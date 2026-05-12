import json
from pathlib import Path

from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

from config import EMBEDDING_MODEL

MANIFEST_FILE = "index_manifest.json"


def _embedding_function():
    return OpenAIEmbeddings(model=EMBEDDING_MODEL)


def _manifest_path(persist_dir):
    return Path(persist_dir) / MANIFEST_FILE


def _load_manifest(persist_dir):
    manifest_path = _manifest_path(persist_dir)
    if not manifest_path.exists():
        return None

    with manifest_path.open("r", encoding="utf-8") as file:
        return json.load(file)


def _save_manifest(persist_dir, manifest):
    manifest_path = _manifest_path(persist_dir)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    with manifest_path.open("w", encoding="utf-8") as file:
        json.dump(manifest, file, indent=2)


def _has_chroma_index(persist_dir):
    return (Path(persist_dir) / "chroma.sqlite3").exists()


def _collection_name(document_fingerprint):
    return f"rag_{document_fingerprint[:16]}"


def is_index_current(document_fingerprint, persist_dir="./chroma_db"):
    manifest = _load_manifest(persist_dir)
    if not manifest or not _has_chroma_index(persist_dir):
        return False

    return (
        manifest.get("document_fingerprint") == document_fingerprint
        and manifest.get("embedding_model") == EMBEDDING_MODEL
        and manifest.get("collection_name") == _collection_name(document_fingerprint)
    )


def load_vectorstore(document_fingerprint, persist_dir="./chroma_db"):
    return Chroma(
        collection_name=_collection_name(document_fingerprint),
        persist_directory=persist_dir,
        embedding_function=_embedding_function()
    )


def rebuild_vectorstore(
    documents,
    document_fingerprint,
    file_count,
    persist_dir="./chroma_db",
):
    collection_name = _collection_name(document_fingerprint)
    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=_embedding_function(),
        collection_name=collection_name,
        persist_directory=persist_dir
    )

    _save_manifest(
        persist_dir,
        {
            "document_fingerprint": document_fingerprint,
            "embedding_model": EMBEDDING_MODEL,
            "collection_name": collection_name,
            "file_count": file_count,
            "chunk_count": len(documents),
        },
    )

    return vectorstore


def get_index_metadata(persist_dir="./chroma_db"):
    return _load_manifest(persist_dir) or {}


def get_vectorstore(documents=None, persist_dir="./chroma_db"):
    embedding = _embedding_function()

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
