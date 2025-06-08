import os
import json
import warnings
import fasttext
import pickle
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from langchain_pinecone import PineconeVectorStore
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
from dotenv import load_dotenv
from src.utils import normalize_text, cluster_questions

# Suppress warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

load_dotenv()

# Set up embeddings and vector store
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
vectorstore = PineconeVectorStore(index_name="university-chatbot", embedding=embeddings)

# Set up LLM
try:
    model_name = "google/flan-t5-base"  # Better for conversational tasks
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    llm_pipeline = pipeline(
        "text2text-generation",
        model=model,
        tokenizer=tokenizer,
        max_length=200,
        device=-1  # CPU; use 0 for GPU
    )
    llm = HuggingFacePipeline(pipeline=llm_pipeline)
except Exception as e:
    raise RuntimeError(f"Failed to load model: {e}")

# Define prompt
prompt = ChatPromptTemplate.from_template(
    "You are a friendly chatbot for a Sri Lankan university. Answer in the same language as the question (Sinhala, Tamil, or English). "
    "For greetings like 'hello' or 'hi', respond conversationally. For specific questions, use the context to provide detailed answers. "
    "Context:\n{context}\n\nQuestion: {question}\nAnswer:"
)

# Set up RAG chain
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
rag_chain = (
    {"context": retriever | (lambda docs: "\n".join([d.page_content for d in docs])), "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# Load fasttext model
try:
    lang_model = fasttext.load_model("lid.176.bin")
except Exception as e:
    raise RuntimeError(f"Failed to load fasttext model: {e}")

# Load or initialize user questions
data_dir = "data"
os.makedirs(data_dir, exist_ok=True)
questions_file = os.path.join(data_dir, "user_questions.json")
try:
    if os.path.exists(questions_file) and os.path.getsize(questions_file) > 0:
        with open(questions_file, "r", encoding="utf-8") as f:
            user_questions = json.load(f)
    else:
        user_questions = []
        with open(questions_file, "w", encoding="utf-8") as f:
            json.dump(user_questions, f, ensure_ascii=False)
except json.JSONDecodeError:
    print("Warning: user_questions.json is corrupted. Initializing empty list.")
    user_questions = []
    with open(questions_file, "w", encoding="utf-8") as f:
        json.dump(user_questions, f, ensure_ascii=False)

# Greeting responses
GREETINGS = {
    "en": ["hello", "hi", "hey"],
    "si": ["ආයුබෝවන්", "ආයුබෝ", "හායි"],
    "ta": ["வணக்கம்", "ஹாய்"]
}
GREETING_RESPONSES = {
    "en": "Hello! How can I assist you today?",
    "si": "ආයුබෝවන්! මා ඔබට අද කෙසේ උපකාර කළ හැකිද?",
    "ta": "வணக்கம்! இன்று உங்களுக்கு எவ்வாறு உதவ முடியும்?"
}

def answer_query(query):
    try:
        # Normalize query
        query = normalize_text(query).lower().strip()
        
        # Detect language
        lang_pred = lang_model.predict(query.replace("\n", " "))[0][0].replace("__label__", "")
        lang_map = {"si": "Sinhala", "ta": "Tamil", "en": "English"}
        lang = lang_map.get(lang_pred, "English")
        lang_code = lang_pred
        print(f"Detected language: {lang} ({lang_pred})")
        
        # Handle greetings
        if any(greet in query for greet in GREETINGS.get(lang_code, [])):
            response = GREETING_RESPONSES.get(lang_code, "Hello! How can I assist you?")
        else:
            # Prioritize positive feedback chunks
            positive_chunks = []
            if os.path.exists(questions_file):
                with open(questions_file, "r", encoding="utf-8") as f:
                    questions = json.load(f)
                positive_queries = [q["query"] for q in questions if q.get("feedback") == "positive"]
                if positive_queries:
                    positive_chunks = vectorstore.similarity_search(" ".join(positive_queries), k=2)
            
            # Combine positive chunks with regular retrieval
            docs = vectorstore.similarity_search(query, k=3)
            combined_context = "\n".join([d.page_content for d in positive_chunks + docs])
            
            # Get response
            response = rag_chain.invoke({"context": combined_context, "question": query})
            response = response.strip() or "No relevant information found."
        
        # Store question and response
        user_questions.append({"query": query, "response": response, "language": lang})
        with open(questions_file, "w", encoding="utf-8") as f:
            json.dump(user_questions, f, ensure_ascii=False)
        
        # Cluster questions every 10 queries
        if len(user_questions) % 10 == 0:
            questions = [q["query"] for q in user_questions]
            labels, kmeans = cluster_questions(questions)
            for cluster_id in range(kmeans.n_clusters):
                cluster_questions = [q for q, l in zip(questions, labels) if l == cluster_id]
                if cluster_questions:
                    pseudo_doc = {"page_content": " ".join(cluster_questions), "metadata": {"source": f"cluster_{cluster_id}"}}
                    vectorstore.add_texts([pseudo_doc["page_content"]], metadatas=[pseudo_doc["metadata"]])
            with open(os.path.join(data_dir, "clusters.pkl"), "wb") as f:
        
        return response
    except Exception as e:
        return f"Error processing query: {e}"

if __name__ == "__main__":
    while True:
        query = input("Enter your question (or 'quit' to exit): ")
        if query.lower() == "quit":
            break
        print(f"Response: {answer_query(query)}\n")