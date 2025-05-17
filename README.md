# Sri Lankan University Chatbot
A trilingual (Sinhala, Tamil, English) chatbot for answering student queries using Pinecone and LangChain.

## Setup
1. Install Python 3.8+ and dependencies: `pip install -r requirements.txt`
2. Set up Pinecone: Create an index named `university-chatbot` (dimension: 384).
3. Add documents to `docs/`.
4. Run `src/process_docs.py` to store embeddings.
5. Test chatbot: `python src/chatbot.py`
6. Launch interface: `streamlit run app.py`

## Environment Variables
- `PINECONE_API_KEY`: Your Pinecone API key.

## Deployment
- Deploy to Streamlit Cloud or a university server.