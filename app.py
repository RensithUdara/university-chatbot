import streamlit as st
from src.chatbot import answer_query

st.title("Sri Lankan University Chatbot")
st.write("Ask questions in Sinhala, Tamil, or English")

# Input box with Unicode support
query = st.text_input("Your Question:", placeholder="Type your question here...")

if st.button("Submit"):
    if query:
        with st.spinner("Generating response..."):
            response = answer_query(query)
            st.write("**Response:**")
            st.write(response)
    else:
        st.error("Please enter a question.")

# Add CSS for Sinhala/Tamil font support
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Noto+Sans+Sinhala&family=Noto+Sans+Tamil&display=swap');
    body, input, textarea {
        font-family: 'Noto Sans Sinhala', 'Noto Sans Tamil', sans-serif;
    }
    </style>
""", unsafe_allow_html=True)