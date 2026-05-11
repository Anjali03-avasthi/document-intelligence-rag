# /// script
# dependencies = [
#   "langchain", "langchain-openai", "langchain-chroma",
#   "pypdf", "python-dotenv", "streamlit", "protobuf==3.20.3"
# ]
# ///

import streamlit as st
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# --- Page Config & Setup ---
st.set_page_config(page_title="RAG PoC Chatbot", page_icon="🤖")
st.title("🤖 Chat with your PDF")
load_dotenv()

# --- Initialize RAG Components (Cached for speed) ---
@st.cache_resource
def initialize_rag():
    # Load and split (make sure the file exists in your directory!)
    loader = PyPDFLoader("sample_report.pdf") 
    docs = loader.load()
    splits = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100).split_documents(docs)
    
    # Store in local ChromaDB
    vectorstore = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings())
    retriever = vectorstore.as_retriever()
    
    # Chain setup
    template = "Answer based on context: {context}\nQuestion: {question}"
    prompt = ChatPromptTemplate.from_template(template)
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
    
    return ({"context": retriever, "question": RunnablePassthrough()} | prompt | llm | StrOutputParser())

rag_chain = initialize_rag()

# --- Chat Interface ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User Input
if prompt := st.chat_input("Ask something about your document..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate Response
    with st.chat_message("assistant"):
        response = rag_chain.invoke(prompt)
        st.markdown(response)
    st.session_state.messages.append({"role": "assistant", "content": response})