from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain.chains import RetrievalQA
from langchain_community.llms import HuggingFacePipeline
from langchain_core.prompts import ChatPromptTemplate
from transformers import pipeline
from langdetect import detect
from dotenv import load_dotenv

load_dotenv()

# Set up embeddings and vector store
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
vectorstore = PineconeVectorStore(index_name="university-chatbot", embedding=embeddings)

# Set up LLM
llm_pipeline = pipeline("text2text-generation", model="google/mt5-small", max_length=200)
llm = HuggingFacePipeline(pipeline=llm_pipeline)

# Define prompt
prompt = ChatPromptTemplate.from_template(
    "You are a chatbot for a Sri Lankan university. Answer in the same language as the question. "
    "Context: {context}\nQuestion: {question}"
)

# Set up QA chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
    chain_type_kwargs={"prompt": prompt}
)

def answer_query(query):
    lang = detect(query)  # Detect language: 'si', 'ta', or 'en'
    response = qa_chain.run(query)
    return response

if __name__ == "__main__":
    queries = [
        "ඉංජිනේරු පීඨයට ඇතුළත් වීමේ අවශ්‍යතා මොනවාද?",  # Sinhala
        "பொறியியல் பீடத்திற்கு விண்ணப்பிப்பது எப்படி?",  # Tamil
        "What are the scholarship options?"  # English
    ]
    for query in queries:
        print(f"Query: {query}")
        print(f"Response: {answer_query(query)}\n")