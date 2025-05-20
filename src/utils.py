import unicodedata
import numpy as np
from sklearn.cluster import KMeans
from langchain_huggingface import HuggingFaceEmbeddings

def normalize_text(text):
       """Normalize Unicode characters for Sinhala/Tamil text."""
       return unicodedata.normalize('NFC', text)

def cluster_questions(questions, n_clusters=5):
       """Cluster user questions for unsupervised learning."""
       embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
       question_embeddings = embeddings.embed_documents(questions)
       kmeans = KMeans(n_clusters=n_clusters, random_state=42)
       labels = kmeans.fit_predict(question_embeddings)
       return labels, kmeans