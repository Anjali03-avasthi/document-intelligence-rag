# Document Intelligence RAG

A Retrieval-Augmented Generation application for asking questions over PDF documents using embeddings, ChromaDB, and an LLM.

The app reads local PDF and CSV files from the `data/` folder, indexes them into a vector database, and lets you ask questions about those documents through a chat interface.

## What This Project Does

This project demonstrates the core RAG flow:

```text
Local PDF/CSV files
-> load documents
-> split text into chunks
-> create embeddings
-> store chunks in Chroma
-> retrieve relevant chunks for a question
-> send retrieved context to an LLM
-> generate an answer with source citations
```

The current app also shows retrieved source chunks, so you can inspect why the model answered the way it did.

## Tech Stack

- Python 3.12
- Streamlit
- LangChain
- OpenAI chat model and embeddings
- Chroma vector database
- PyPDF for PDF loading
- CSV document loading through LangChain

## Project Structure

```text
.
├── src/
│   ├── app.py          # Streamlit UI
│   ├── loader.py       # Loads PDF/CSV files and splits documents
│   ├── database.py     # Creates or loads the Chroma vector store
│   └── chain.py        # RAG prompt, retrieval, answer generation, citations
├── data/
│   └── .gitkeep        # Put your local PDFs and CSVs here
├── rag_poc.py          # Earlier single-file proof of concept
├── requirements.txt
├── pyproject.toml
└── README.md
```

## Setup

Clone the repository:

```bash
git clone https://github.com/singhbad/semantic-search-rag.git
cd semantic-search-rag
```

Install dependencies with `uv`:

```bash
uv sync
```

Or install from `requirements.txt`:

```bash
pip install -r requirements.txt
```

Create a `.env` file:

```bash
touch .env
```

Add your OpenAI API key:

```env
OPENAI_API_KEY=your_openai_api_key_here
```

## Add Your Documents

Place your local documents inside the `data/` folder:

```text
data/
├── report.pdf
├── notes.pdf
└── records.csv
```

The app searches recursively inside `data/`, so nested folders also work:

```text
data/company_reports/report_2023.pdf
```

Local documents are ignored by Git, so private files are not pushed to GitHub.

## Run The App

```bash
uv run streamlit run src/app.py
```

Or, if you installed dependencies with `pip`:

```bash
streamlit run src/app.py
```

Then open:

```text
http://localhost:8501
```

## Example Questions

Ask questions based on your local files:

```text
What numbers of Wood used for fuelling boilers in tons for 2021 and 2020?
```

```text
Summarize the ESG performance in the uploaded reports.
```

```text
Which source mentions emissions reduction targets?
```

## Learning Notes

This is a beginner RAG project. It currently teaches:

- document loading
- text chunking
- embeddings
- vector search
- retrieval
- prompt construction
- grounded answers
- source citations
- Streamlit UI basics

Good next improvements:

- make retrieval `k` configurable from the sidebar
- show full retrieved chunks in the UI
- add page-level citations in the final answer
- avoid re-indexing documents that are already stored
- add support for more file types
- evaluate answer quality with test questions

## Important Git Notes

The following are intentionally ignored:

- `.env`
- `.venv/`
- `chroma_db/`
- local files inside `data/`
- temporary PDF files

This keeps secrets, private documents, and generated vector database files out of the remote repository.
