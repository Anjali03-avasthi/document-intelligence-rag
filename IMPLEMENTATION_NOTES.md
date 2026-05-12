# RAG POC Implementation Notes

This file records each improvement made to the POC so the implementation choices are easy to revisit later.

## 1. Explicit Model Configuration

### Goal

Make the embedding model and answering model explicit instead of relying on hidden defaults or hardcoded values spread across multiple files.

### Changed Files

- `src/config.py`
- `src/database.py`
- `src/chain.py`

### Implementation

Created `src/config.py`:

```python
EMBEDDING_MODEL = "text-embedding-3-small"
ANSWER_MODEL = "gpt-4o-mini"
```

Updated `src/database.py` so Chroma embeddings use:

```python
OpenAIEmbeddings(model=EMBEDDING_MODEL)
```

Updated `src/chain.py` so the answer generation model uses:

```python
ChatOpenAI(model=ANSWER_MODEL, temperature=0)
```

### Why This Helps

- The POC now has a single place to inspect or change model choices.
- The embedding model is pinned, so behavior is less dependent on LangChain/OpenAI defaults.
- Documentation and code now agree on which models are being used.
- Future upgrades, such as switching to `text-embedding-3-large` or a newer answer model, only require changing `src/config.py`.

### Current Models

- Embeddings: `text-embedding-3-small`
- Answering: `gpt-4o-mini`

### Next Improvement

Add persistent index loading so the app does not rebuild the Chroma index from scratch every time it starts.

## 2. Persistent Chroma Index Loading

### Goal

Avoid rebuilding embeddings every time the Streamlit app starts when the local PDF/CSV files have not changed.

### Changed Files

- `src/loader.py`
- `src/database.py`
- `src/app.py`
- `IMPLEMENTATION_NOTES.md`

### Implementation

Added file discovery and fingerprinting in `src/loader.py`:

```python
list_supported_files(data_dir)
fingerprint_files(files, data_dir)
```

The fingerprint is based on each supported file's relative path, file size, and last modified timestamp.

Added Chroma index management in `src/database.py`:

```python
is_index_current(document_fingerprint)
load_vectorstore(document_fingerprint)
rebuild_vectorstore(...)
get_index_metadata()
```

When rebuilding, the app writes a manifest file at:

```text
chroma_db/index_manifest.json
```

The manifest stores:

- document fingerprint
- embedding model
- Chroma collection name
- file count
- chunk count

Updated `src/app.py` startup flow:

1. Find supported files in `data/`.
2. Stop early with a warning if no PDF/CSV files are present.
3. Compute a document fingerprint.
4. Load the existing Chroma index if the fingerprint and embedding model match.
5. Rebuild the Chroma index only when files or the embedding model changed.

### Why This Helps

- Startup is faster after the first successful index build.
- The app avoids repeated embedding API calls for unchanged documents.
- Changing the embedding model automatically triggers a rebuild.
- The sidebar now shows whether the app loaded or rebuilt the index.

### Rebuild Triggers

The index rebuilds when:

- a supported file is added
- a supported file is removed
- a supported file's size or modified timestamp changes
- `EMBEDDING_MODEL` changes in `src/config.py`

### Current Limitation

This is still a full rebuild when anything changes. To avoid SQLite write errors from deleting a database while Streamlit/Chroma may still have an open handle, rebuilds create a new Chroma collection named from the current document fingerprint instead of deleting and recreating the whole database directory.

A future improvement could update only the changed files and periodically clean up old fingerprint-based collections.

## 3. Streamlit File Upload

### Goal

Let users add PDF and CSV files directly from the Streamlit UI instead of manually copying files into the `data/` folder.

### Changed Files

- `src/app.py`
- `IMPLEMENTATION_NOTES.md`

### Implementation

Added a sidebar uploader in `src/app.py`:

```python
st.sidebar.file_uploader(
    "Upload PDF or CSV files",
    type=["pdf", "csv"],
    accept_multiple_files=True,
)
```

Added a helper that saves uploaded files into the app's existing `data/` directory:

```python
def save_uploaded_files(uploaded_files):
    ...
    destination = data_dir / uploaded_file.name
    destination.write_bytes(uploaded_file.getbuffer())
```

After saving files, the app calls:

```python
st.rerun()
```

On rerun, the persistent-index logic detects the changed file fingerprint and rebuilds the Chroma index.

### Why This Helps

- The app is easier to demo because documents can be added through the browser.
- Uploaded files use the same ingestion path as manually added files.
- The existing fingerprint and manifest logic handles index rebuilding automatically.

### Current Behavior

- Supported upload types: PDF and CSV.
- Uploaded files are saved into `data/`.
- If an uploaded file has the same name as an existing file, it overwrites the existing file.
- The app reruns after upload so the changed document set is indexed.

### Current Limitation

There is no duplicate-file warning yet. A future improvement could detect existing file names and ask whether to overwrite, rename, or skip.

## 4. Basic RAG Evaluation Script

### Goal

Add a repeatable way to check whether the RAG system retrieves useful sources and includes expected terms in answers.

### Changed Files

- `eval_questions.json`
- `eval_rag.py`
- `IMPLEMENTATION_NOTES.md`

### Implementation

Created `eval_questions.json` with test cases:

```json
{
  "question": "What numbers of wood used for fuelling boilers in tons for 2021 and 2020?",
  "expected_sources": [
    "2022-Absa-Group-limited-Environmental-Social-and-Governance-Data-sheet.pdf"
  ],
  "expected_terms": ["wood", "2021", "2020", "tons"]
}
```

Created `eval_rag.py`, which:

1. Loads eval questions from JSON.
2. Loads environment variables from `.env`.
3. Loads the existing Chroma index if it is current.
4. Rebuilds the index if the document fingerprint changed.
5. Runs each question through the same RAG chain used by the Streamlit app.
6. Prints retrieved sources, answer text, and simple PASS/FAIL checks.

Run it with:

```bash
uv run python eval_rag.py
```

Or:

```bash
python3 eval_rag.py
```

Optional arguments:

```bash
python3 eval_rag.py --eval-file eval_questions.json --data-dir data --k 5
```

### What PASS Means

A question passes when:

- every `expected_sources` entry appears in the retrieved source list
- every `expected_terms` entry appears in the final answer text

### Why This Helps

- It gives a repeatable test for retrieval quality.
- It helps compare changes to chunk size, retrieval `k`, search type, or model choice.
- It catches cases where the app gives an answer but retrieves the wrong source.

### Current Limitation

This eval is a simple smoke test, not a full judge. It does not know whether the answer is logically complete or numerically correct beyond checking expected terms. A future improvement could add LLM-as-judge evaluation or exact expected-answer checks.
