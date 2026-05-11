from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

def get_vectorstore(documents=None, persist_dir="./chroma_db"):
    embedding = OpenAIEmbeddings()

    if documents:
        return Chroma.from_documents(
            documents=documents,
            embedding=embedding,
            persist_directory=persist_dir
        )
    else:
        return Chroma(
            persist_directory=persist_dir,
            embedding_function=embedding
        )
