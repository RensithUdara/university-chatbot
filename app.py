import streamlit as st
import json
import os
from src.chatbot import answer_query

st.title("Sri Lankan University Chatbot")
st.write("Ask questions in Sinhala, Tamil, or English. I'm here to help with university information or just chat!")

# Welcome message
st.markdown("**Welcome!** Type 'hello' or ask about admissions, courses, or scholarships.")

# Display recent questions
if os.path.exists("data/user_questions.json"):
    with open("data/user_questions.json", "r", encoding="utf-8") as f:
        questions = json.load(f)
    if questions:
        st.subheader("Recent Questions")
        for q in questions[-3:]:
            st.write(f"**Q:** {q['query']} **A:** {q['response']}")

# Input box
query = st.text_input("Your Question:", placeholder="E.g., 'hello' or 'What are the scholarship options?'")

if st.button("Submit"):
    if query:
        with st.spinner("Generating response..."):
            response = answer_query(query)
            st.write("**Response:**")
            st.write(response)
            
            # Feedback mechanism
            st.write("Was this response helpful?")
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Yes"):
                    with open("data/user_questions.json", "r", encoding="utf-8") as f:
                        questions = json.load(f)
                    questions[-1]["feedback"] = "positive"
                    with open("data/user_questions.json", "w", encoding="utf-8") as f:
                        json.dump(questions, f, ensure_ascii=False)
                    st.success("Thank you for your feedback!")
            with col2:
                if st.button("No"):
                    with open("data/user_questions.json", "r", encoding="utf-8") as f:
                        questions = json.load(f)
                    questions[-1]["feedback"] = "negative"
                    with open("data/user_questions.json", "w", encoding="utf-8") as f:
                        json.dump(questions, f, ensure_ascii=False)
                    st.success("Thank you for your feedback!")
    else:
        st.error("Please enter a question.")

# Font support for Sinhala/Tamil
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Noto+Sans+Sinhala&family=Noto+Sans+Tamil&display=swap');
    body, input, textarea {
        font-family: 'Noto Sans Sinhala', 'Noto Sans Tamil', sans-serif;
    }
    </style>
""", unsafe_allow_html=True)