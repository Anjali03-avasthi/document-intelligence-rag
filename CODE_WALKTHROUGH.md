# Code Walkthrough — How to Read This Project

Read the files in this exact order. Each one builds on the previous.
Do not jump ahead — the mental model stacks up layer by layer.

---

## Reading Order

### 1. `src/config.py` — 2 lines, start here

The simplest file. Two constants that every other file depends on.

```python
EMBEDDING_MODEL = "text-embedding-3-small"
ANSWER_MODEL = "gpt-4o-mini"
```

**Understand:** Why are model names centralised here instead of hardcoded in each file?
Answer: change one line → the whole system switches models. Without this, you would
have to hunt down every file that mentions a model name.

---

### 2. `src/loader.py` — the data entry point

This is where raw files become searchable chunks. Read these three functions in order:

| Function | What it does |
|----------|-------------|
| `list_supported_files()` | Scans `data/` and returns paths of all PDFs and CSVs |
| `fingerprint_files()` | SHA-256 hash of each file's path + size + modified time |
| `ingest_local_documents()` | Loads files → splits into chunks → returns `List[Document]` |

**Understand:** How does a 50-page PDF become 200 small text pieces?
Answer: `RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)` cuts the
text into overlapping windows. Each window becomes one `Document` object with
`page_content` (the text) and `metadata` (source filename, page number).

---

### 3. `src/database.py` — the index layer

Read top to bottom. Focus on these four functions:

| Function | What it does |
|----------|-------------|
| `is_index_current()` | Compares live fingerprint to stored manifest — rebuild needed? |
| `rebuild_vectorstore()` | Embeds chunks via OpenAI, stores vectors in ChromaDB, writes manifest |
| `load_vectorstore()` | Loads existing ChromaDB index without re-embedding |
| `get_index_metadata()` | Reads the manifest (chunk count, model used, file count) |

**Understand:** Why does the manifest store the embedding model name?
Answer: If you switch from `text-embedding-3-small` to `text-embedding-3-large`,
the old vectors are in a completely different vector space. Comparing them would give
nonsense results. Storing the model name lets the app detect this and trigger a rebuild.

---

### 4. `src/chain.py` — the brain

The most important file. Read it in the order the pieces are defined:

**Step 1 — History store**
```python
_session_store: dict[str, ChatMessageHistory] = {}

def _get_session_history(session_id):
    ...
```
A plain Python dict keyed by session ID. Each value is a list of past messages.
`RunnableWithMessageHistory` calls this automatically — you never touch it directly.

**Step 2 — History-aware retrieval**
```python
def history_aware_retrieve(inputs):
    if inputs.get("chat_history"):
        standalone_q = contextualize_chain.invoke(inputs)
    else:
        standalone_q = inputs["input"]
    return retriever.invoke(standalone_q)
```
Follow-up questions like "tell me more about that" contain no useful keywords for
vector search. This function rewrites them into standalone questions first.
Example: "tell me more about that" → "tell me more about the 2022 emissions targets".

**Step 3 — Answer chain**
```python
answer_chain = (
    {
        "context": lambda x: "\n\n".join(d.page_content for d in x["context"]),
        "input": lambda x: x["input"],
        "chat_history": lambda x: x.get("chat_history", []),
    }
    | qa_prompt
    | llm
    | StrOutputParser()
)
```
Takes the full state dict, formats the retrieved documents into a string, builds
the prompt, calls the LLM, and parses the output. The `|` pipes are LCEL — each
step passes its output to the next. This chain streams tokens.

**Step 4 — Full RAG chain**
```python
rag_chain = (
    RunnablePassthrough.assign(context=RunnableLambda(history_aware_retrieve))
    .assign(answer=answer_chain)
)
```
Two `.assign()` calls build up the state dict:
- After first assign: `{input, chat_history, context: List[Document]}`
- After second assign: `{input, chat_history, context, answer: str (streaming)}`

**Step 5 — Memory wrapper**
```python
return RunnableWithMessageHistory(
    rag_chain,
    _get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
    output_messages_key="answer",
)
```
Wraps the chain so every `.invoke()` or `.stream()` call automatically loads
history before running and saves the new exchange after running.

**Understand:** How does a question travel through this chain?
```
User question
      │
      ▼
_get_session_history()        ← load past messages for this session_id
      │
      ▼
history_aware_retrieve()      ← rewrite question if history exists
      │
      ▼
ChromaDB vector search        ← top-K relevant chunks returned
      │
      ▼
answer_chain                  ← format docs + build prompt + call LLM
      │
      ▼
LLM streams tokens            ← each token is a {"answer": "token"} chunk
      │
      ▼
_get_session_history()        ← save (human message, ai message) to history
```

---

### 5. `src/app.py` — the glue

Read this last. It calls everything above. Focus on three parts:

**Index initialisation**
```python
@st.cache_resource
def initialize_index(cache_version, document_fingerprint):
    if is_index_current(document_fingerprint):
        return load_vectorstore(...)       # fast path
    return rebuild_vectorstore(...)        # slow path — re-embeds everything
```
`@st.cache_resource` means this only runs once per unique `(cache_version, fingerprint)`
combination. Upload a new file → fingerprint changes → Streamlit re-runs this.

**Streaming loop**
```python
for chunk in chain.stream({"input": prompt}, config={"configurable": {"session_id": CHAT_SESSION_ID}}):
    if "context" in chunk and not retrieved_sources:
        retrieved_sources = source_list(chunk["context"])   # arrives first
    if "answer" in chunk:
        answer_text += chunk["answer"]
        placeholder.markdown(answer_text + "▌")             # live cursor
```
The `context` chunk arrives once (retrieval done). Then `answer` chunks stream
one token at a time. Sources are shown while the answer is still being written.

**Two parallel history records**
```python
st.session_state.messages   # Streamlit's display history (what the UI renders)
_session_store              # LangChain's chain history (what the LLM sees)
```
These are kept in sync: every user message and assistant response is appended
to both. `session_state.messages` is for rendering the chat UI; `_session_store`
is for the LLM's context. They contain the same information in different formats.

---

### 6. `eval_rag.py` — outside the UI stack

Sits parallel to the app, not inside it. Imports `chain.py` and `database.py` directly.

```python
result = chain.invoke(
    {"input": question},
    config={"configurable": {"session_id": f"eval_{index}"}},
)
```

Each question gets its own session ID so eval questions do not bleed into each
other's history. After getting the result, it runs two checks:
- **Keyword check** — deterministic, fast, good for regression tests
- **LLM judge** — semantic, catches hallucinations and paraphrases

---

## The Full Mental Map

```
config.py
  └── model names (EMBEDDING_MODEL, ANSWER_MODEL)
        │
        ├── loader.py
        │     └── PDF/CSV → List[Document] (chunks with metadata)
        │                         │
        │                         ▼
        ├── database.py
        │     └── List[Document] → ChromaDB vectors
        │         fingerprint → manifest → skip rebuild if unchanged
        │                         │
        │                         ▼ (retriever object)
        └── chain.py
              └── retriever + LLM + memory → RunnableWithMessageHistory
                                    │
                    ┌───────────────┴───────────────┐
                    │                               │
              app.py                          eval_rag.py
         Streamlit UI                     CLI evaluation script
         .stream() → live tokens          .invoke() → score answers
```

---

## One-line summary of each file

| File | One line |
|------|----------|
| `src/config.py` | Model names in one place |
| `src/loader.py` | Files → chunks |
| `src/database.py` | Chunks → vectors, cached by fingerprint |
| `src/chain.py` | Retriever + memory + LLM → streaming chain |
| `src/app.py` | Wires everything, renders in browser |
| `eval_rag.py` | Runs questions through the chain, scores answers |

---

*Last updated: 2026-05-12*
