import os
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from dotenv import load_dotenv

load_dotenv()

# Load documents
loader = PyPDFDirectoryLoader("docs/")
docs = loader.load()

# Split into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
chunks = text_splitter.split_documents(docs)

# Save chunks for debugging
with open("chunks.txt", "w", encoding="utf-8") as f:
    for i, chunk in enumerate(chunks):
        f.write(f"Chunk {i}:\n{chunk.page_content}\n\n")

# Set up embeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

# Store in Pinecone
index_name = "university-chatbot"
vectorstore = PineconeVectorStore.from_documents(chunks, embeddings, index_name=index_name)

print("Documents processed and stored in Pinecone.")