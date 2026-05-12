# /// script
# dependencies = [
#   "langchain", "langchain-openai", "langchain-chroma", "langchain-community",
#   "pypdf", "python-dotenv", "streamlit", "protobuf==3.20.3"
# ]
# ///

from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from loader import fingerprint_files, ingest_local_documents, list_supported_files
from database import (
    get_index_metadata,
    is_index_current,
    load_vectorstore,
    rebuild_vectorstore,
)
from chain import create_rag_chain, source_list

load_dotenv()

st.title("Modular RAG Explorer")

DATA_DIR = "data"
RAG_CACHE_VERSION = "retrieval-debug-v1"
# A fixed session ID ties LangChain's in-memory history to this browser session.
# In a multi-user app you would derive this from the authenticated user or a UUID
# stored in st.session_state.
CHAT_SESSION_ID = "streamlit"


def save_uploaded_files(uploaded_files):
    saved_files = []
    data_dir = Path(DATA_DIR)
    data_dir.mkdir(parents=True, exist_ok=True)

    for uploaded_file in uploaded_files:
        destination = data_dir / uploaded_file.name
        destination.write_bytes(uploaded_file.getbuffer())
        saved_files.append(destination)

    return saved_files


@st.cache_resource
def initialize_index(cache_version, document_fingerprint):
    files = list_supported_files(DATA_DIR)

    if is_index_current(document_fingerprint):
        metadata = get_index_metadata()
        vectorstore = load_vectorstore(document_fingerprint)
        return (
            vectorstore,
            files,
            metadata.get("chunk_count", 0),
            "Loaded existing Chroma index",
        )

    docs, files = ingest_local_documents(DATA_DIR)
    vectorstore = rebuild_vectorstore(
        documents=docs,
        document_fingerprint=document_fingerprint,
        file_count=len(files),
    )

    return vectorstore, files, len(docs), "Rebuilt Chroma index"


def display_sources(sources):
    with st.expander("Retrieved sources"):
        for source in sources:
            st.markdown(f"**[{source['id']}] {source['source']}**")
            st.caption(source["preview"])
            with st.expander(f"Full chunk [{source['id']}]", expanded=False):
                st.write(source["content"])


uploaded_files = st.sidebar.file_uploader(
    "Upload PDF or CSV files",
    type=["pdf", "csv"],
    accept_multiple_files=True,
)

if uploaded_files and st.sidebar.button("Save uploaded files"):
    saved_files = save_uploaded_files(uploaded_files)
    st.sidebar.success(f"Saved {len(saved_files)} file(s). Rebuilding index...")
    st.rerun()

index_files = list_supported_files(DATA_DIR)
if not index_files:
    st.warning(f"Add PDF or CSV files to `{DATA_DIR}/` to start chatting.")
    st.stop()

document_fingerprint = fingerprint_files(index_files, DATA_DIR)

with st.status("Preparing local PDF and CSV index...", expanded=False):
    vectorstore, files, chunk_count, index_status = initialize_index(
        RAG_CACHE_VERSION,
        document_fingerprint,
    )
    st.write(index_status)

retrieval_k = st.sidebar.slider("Retrieved chunks", min_value=1, max_value=15, value=5)
search_type = st.sidebar.selectbox("Search type", ["similarity", "mmr"])

search_kwargs = {"k": retrieval_k}
if search_type == "mmr":
    search_kwargs = {"k": retrieval_k, "fetch_k": max(20, retrieval_k * 3)}

retriever = vectorstore.as_retriever(
    search_type=search_type,
    search_kwargs=search_kwargs,
)
chain = create_rag_chain(retriever)

st.sidebar.write(f"Loaded `{len(files)}` files from `{DATA_DIR}/`")
st.sidebar.write(f"Created `{chunk_count}` searchable chunks")
st.sidebar.caption(index_status)
for file_path in files:
    st.sidebar.caption(str(file_path))

st.success("Ready to chat with your local files.")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        if msg.get("sources"):
            display_sources(msg["sources"])

if prompt := st.chat_input("Ask a question about your local PDF/CSV files..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    # Stream the answer token-by-token so the UI is responsive.
    # How streaming works with create_retrieval_chain:
    #   1. First chunk contains {"context": [Document, ...]} — retrieval is done.
    #   2. Subsequent chunks contain {"answer": "token"} — LLM is generating.
    # We grab sources from chunk 1, then accumulate answer tokens as they arrive.
    retrieved_sources = []
    answer_text = ""

    with st.chat_message("assistant"):
        placeholder = st.empty()

        for chunk in chain.stream(
            {"input": prompt},
            config={"configurable": {"session_id": CHAT_SESSION_ID}},
        ):
            if "context" in chunk and not retrieved_sources:
                retrieved_sources = source_list(chunk["context"])
            if "answer" in chunk:
                answer_text += chunk["answer"]
                # ▌ acts as a typing cursor so the user sees generation is live.
                placeholder.markdown(answer_text + "▌")

        placeholder.markdown(answer_text)
        display_sources(retrieved_sources)

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer_text,
            "sources": retrieved_sources,
        }
    )
