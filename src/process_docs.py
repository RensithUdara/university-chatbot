import os
import json
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from dotenv import load_dotenv
from src.utils import normalize_text

load_dotenv()

# Load documents
loader = PyPDFDirectoryLoader("docs/")
docs = loader.load()

# Normalize text for Sinhala/Tamil
for doc in docs:
    doc.page_content = normalize_text(doc.page_content)
    # Add metadata (e.g., file name, page number)
    doc.metadata["source"] = doc.metadata.get("source", "unknown")
    doc.metadata["page"] = doc.metadata.get("page", 0)

# Split into smaller chunks for precise retrieval
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = text_splitter.split_documents(docs)

# Save chunks with metadata for debugging
with open("chunks.txt", "w", encoding="utf-8") as f:
    for i, chunk in enumerate(chunks):
        f.write(f"Chunk {i}:\nContent: {chunk.page_content}\nMetadata: {chunk.metadata}\n\n")

# Set up embeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

# Store in Pinecone with metadata
index_name = "university-chatbot"
vectorstore = PineconeVectorStore.from_documents(chunks, embeddings, index_name=index_name)

# Save chunk metadata for unsupervised learning
with open("data/chunk_metadata.json", "w", encoding="utf-8") as f:
    json.dump([{"id": i, "content": c.page_content, "metadata": c.metadata} for i, c in enumerate(chunks)], f, ensure_ascii=False)

print("Documents processed and stored in Pinecone.")