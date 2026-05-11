# /// script
# dependencies = [
#   "langchain", "langchain-openai", "langchain-chroma",
#   "pypdf", "python-dotenv", "streamlit", "protobuf==3.20.3"
# ]
# ///

import streamlit as st
from dotenv import load_dotenv
from loader import ingest_local_documents
from database import get_vectorstore
from chain import create_rag_chain

load_dotenv()

st.title("Modular RAG Explorer")

DATA_DIR = "data"
RAG_CACHE_VERSION = "citations-v1"


@st.cache_resource
def initialize_rag(cache_version):
    docs, files = ingest_local_documents(DATA_DIR)
    vectorstore = get_vectorstore(docs)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 10})
    chain = create_rag_chain(retriever)
    return chain, files, len(docs)


with st.status("Indexing local PDF and CSV files...", expanded=False):
    chain, files, chunk_count = initialize_rag(RAG_CACHE_VERSION)

st.sidebar.write(f"Loaded `{len(files)}` files from `{DATA_DIR}/`")
st.sidebar.write(f"Created `{chunk_count}` searchable chunks")
for file_path in files:
    st.sidebar.caption(str(file_path))

st.success("Ready to chat with your local files.")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if prompt := st.chat_input("Ask a question about your local PDF/CSV files..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    result = chain(prompt)
    response = result["answer"]

    st.session_state.messages.append({"role": "assistant", "content": response})
    with st.chat_message("assistant"):
        st.write(response)

        with st.expander("Retrieved sources"):
            for source in result["sources"]:
                st.markdown(f"**[{source['id']}] {source['source']}**")
                st.caption(source["preview"])
