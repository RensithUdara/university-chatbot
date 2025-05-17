import os
import warnings
import fasttext
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_huggingface import HuggingFacePipeline
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
from dotenv import load_dotenv

# Suppress warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

load_dotenv()

# Set up embeddings and vector store
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
vectorstore = PineconeVectorStore(index_name="university-chatbot", embedding=embeddings)

# Set up LLM
try:
    model_name = "google/mt5-small"
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True, legacy=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    llm_pipeline = pipeline(
        "text2text-generation",
        model=model,
        tokenizer=tokenizer,
        max_length=200,
        device=-1  # CPU; use 0 for GPU if available
    )
    llm = HuggingFacePipeline(pipeline=llm_pipeline)
except Exception as e:
    raise RuntimeError(f"Failed to load model: {e}")

# Define prompt (adjusted for mT5)
prompt = ChatPromptTemplate.from_template(
    "You are a chatbot for a Sri Lankan university. Answer in the same language as the question (Sinhala, Tamil, or English). "
    "Use the following context to provide an accurate response:\n{context}\n\nQuestion: {question}\nAnswer:"
)

# Set up RAG chain
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
rag_chain = (
    {"context": retriever | (lambda docs: "\n".join([d.page_content for d in docs])), "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# Load fasttext model for language detection
try:
    lang_model = fasttext.load_model("lid.176.bin")
except Exception as e:
    raise RuntimeError(f"Failed to load fasttext model: {e}")

def answer_query(query):
    try:
        # Detect language
        lang_pred = lang_model.predict(query.replace("\n", " "))[0][0].replace("__label__", "")
        lang_map = {"si": "Sinhala", "ta": "Tamil", "en": "English"}
        lang = lang_map.get(lang_pred, "English")
        # Log detected language for debugging
        print(f"Detected language: {lang} ({lang_pred})")
        response = rag_chain.invoke(query)
        return response.strip() or "No relevant information found."
    except Exception as e:
        return f"Error processing query: {e}"

if __name__ == "__main__":
    queries = [
        "හෙලෝ",  # Sinhala
        "பொறியியல் பீடத்திற்கு விண்ணப்பிப்பது எப்படி?",  # Tamil
        "What is the Open University?"  # English
    ]
    for query in queries:
        print(f"Query: {query}")
        print(f"Response: {answer_query(query)}\n")