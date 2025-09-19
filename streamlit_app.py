# streamlit_app.py
import os
import streamlit as st
import pandas as pd
from PyPDF2 import PdfReader

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from transformers import pipeline
from langchain_community.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA
from langchain.memory import ConversationBufferWindowMemory

# ----------------------
# Utility Functions
# ----------------------

def load_pdf(file):
    reader = PdfReader(file)
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"
    return text

def load_excel(file):
    try:
        df = pd.read_excel(file)
    except Exception:
        df = pd.read_csv(file)
    return df.to_string()

def build_vectorstore(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_text(text)
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = FAISS.from_texts(chunks, embeddings)
    return vectorstore

def build_qa_chain(vectorstore):

    pipe = pipeline(
        "text2text-generation",
        model="google/flan-t5-base",
        max_length=512,
        temperature=0.2
    )
    llm = HuggingFacePipeline(pipeline=pipe)

    memory = ConversationBufferWindowMemory(k=2, memory_key="chat_history", return_messages=True)

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return qa_chain


# ----------------------
# Streamlit App
# ----------------------

st.set_page_config(page_title="Financial Report QnA", layout="wide")
st.title("📊 Financial Report Q&A Chatbot (Ollama + HuggingFace)")

uploaded_file = st.file_uploader("Upload Financial Report (PDF/Excel)", type=["pdf", "xlsx", "xls", "csv"])

if uploaded_file:
    file_extension = uploaded_file.name.split(".")[-1].lower()
    if file_extension == "pdf":
        text = load_pdf(uploaded_file)
    else:
        text = load_excel(uploaded_file)

    st.success("File loaded successfully!")

    vectorstore = build_vectorstore(text)
    qa_chain = build_qa_chain(vectorstore)

    if qa_chain:
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []

        query = st.chat_input("Ask a question about the report...")
        if query:
            response = qa_chain.run(query)
            st.session_state.chat_history.append((query, response))

        for q, a in st.session_state.chat_history:
            st.chat_message("user").write(q)
            st.chat_message("assistant").write(a)
else:
    st.info("Please upload a financial statement (PDF or Excel) to begin.")
