import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader
import time
from dotenv import load_dotenv

load_dotenv()

## Load the Groq API
os.environ['GROQ_API_KEY'] = os.getenv("GROQ_API_KEY1")
GROQ_API_KEY1 = os.getenv("GROQ_API_KEY1")

llm = ChatGroq(model_name="mixtral-8x7b-32768", temperature=0)  # Removed incorrect GROQ_API_KEY argument

prompt = ChatPromptTemplate.from_template(
    """
    Answer the questions based on the provided context only.
    Please provide the most accurate response based on the question.
    <context>
    {context}
    <context>
    question: {input}
    """
) 

def create_vector_embedding():
    if "vectorstore" not in st.session_state:
        st.session_state.embeddings = OllamaEmbeddings()  # Instantiate the embeddings
        st.session_state.loader = PyPDFDirectoryLoader("Research_papers")  # Data ingestion step
        st.session_state.docs = st.session_state.loader.load()  # Document loader
        st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs[:50])
        st.session_state.vectorstore = FAISS.from_documents(st.session_state.final_documents, st.session_state.embeddings)

st.title("RAG Document Q&A with Groq and Llama")

user_prompt = st.text_input("Enter your query from the research paper")

if st.button("Document Embedding"):
    create_vector_embedding()
    st.write("Vector database is ready")

if user_prompt:
    if "vectorstore" not in st.session_state:
        st.write("Please generate the vector database first.")
    else:
        documents_chain = create_stuff_documents_chain(llm, prompt)
        retriever = st.session_state.vectorstore.as_retriever()
        retrieval_chain = create_retrieval_chain(retriever, documents_chain)
        
        start_time = time.process_time()  # Corrected variable name
        response = retrieval_chain.invoke({"input": user_prompt})
        end_time = time.process_time()  # Corrected time tracking

        st.write(f"Response time: {end_time - start_time:.2f} seconds")
        st.write(response["answer"])
        
        ## Write a Streamlit expander
        with st.expander("Document similarity search"):
            for i, doc in enumerate(response["context"]):
                st.write(doc.page_content)
                st.write("---------------------------------")
