from pathlib import Path

from langchain_community.document_loaders import CSVLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

SUPPORTED_EXTENSIONS = {".pdf", ".csv"}


def _load_file(file_path: Path):
    if file_path.suffix.lower() == ".pdf":
        return PyPDFLoader(str(file_path)).load()

    if file_path.suffix.lower() == ".csv":
        return CSVLoader(str(file_path)).load()

    return []


def ingest_documents(file_path: str):
    data = _load_file(Path(file_path))
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    splits = text_splitter.split_documents(data)

    return splits


def ingest_local_documents(data_dir: str = "data"):
    data_path = Path(data_dir)
    files = [
        path
        for path in data_path.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    ]

    documents = []
    for file_path in files:
        documents.extend(_load_file(file_path))

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    splits = text_splitter.split_documents(documents)

    return splits, files
