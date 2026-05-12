from pathlib import Path
import hashlib

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
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    splits = text_splitter.split_documents(data)

    return splits


def list_supported_files(data_dir: str = "data"):
    data_path = Path(data_dir)
    return [
        path
        for path in data_path.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    ]


def fingerprint_files(files, data_dir: str = "data"):
    data_path = Path(data_dir)
    fingerprint = hashlib.sha256()

    for file_path in sorted(files):
        stats = file_path.stat()
        relative_path = file_path.relative_to(data_path)
        fingerprint.update(str(relative_path).encode("utf-8"))
        fingerprint.update(str(stats.st_size).encode("utf-8"))
        fingerprint.update(str(stats.st_mtime_ns).encode("utf-8"))

    return fingerprint.hexdigest()


def ingest_local_documents(data_dir: str = "data"):
    files = list_supported_files(data_dir)

    documents = []
    for file_path in files:
        documents.extend(_load_file(file_path))

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    splits = text_splitter.split_documents(documents)

    return splits, files
