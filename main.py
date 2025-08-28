import streamlit as st 
from langchain_coding import get_QA_Chain,create_vector_DB

st.title("Nira AI Q&A...")
button= st.button("Create Knowlege Base")

if button:
    create_vector_DB()

question= st.text_input("Question:  ")

if question:
    chain=get_QA_Chain()
    response= chain(question)

    st.header("Answer")
    st.write(response["results"])