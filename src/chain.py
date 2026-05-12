from pathlib import Path

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory

from config import ANSWER_MODEL

# In-memory conversation store: session_id → ChatMessageHistory.
# Lives as long as the Python process; survives Streamlit reruns within a session.
_session_store: dict[str, ChatMessageHistory] = {}


def _get_session_history(session_id: str) -> ChatMessageHistory:
    if session_id not in _session_store:
        _session_store[session_id] = ChatMessageHistory()
    return _session_store[session_id]


def _source_name(document):
    source = document.metadata.get("source", "Unknown source")
    name = Path(source).name
    if "page" in document.metadata:
        return f"{name}, page {document.metadata['page'] + 1}"
    if "row" in document.metadata:
        return f"{name}, row {document.metadata['row']}"
    return name


def source_list(documents):
    """Convert LangChain Document objects into display-ready dicts."""
    return [
        {
            "id": i,
            "source": _source_name(doc),
            "preview": doc.page_content[:400],
            "content": doc.page_content,
        }
        for i, doc in enumerate(documents, 1)
    ]


def create_rag_chain(retriever):
    llm = ChatOpenAI(model=ANSWER_MODEL, temperature=0)

    # ── Step 1: Rewrite follow-up questions before retrieval ─────────────────
    # "tell me more about that" → "tell me more about the emissions targets"
    # Without this, the vector store gets vague pronouns instead of real terms
    # and retrieves irrelevant chunks.
    contextualize_q_prompt = ChatPromptTemplate.from_messages([
        ("system", (
            "Given the conversation history and the latest user question, "
            "rewrite it as a self-contained question. "
            "Do NOT answer it — only rewrite or return it unchanged."
        )),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    contextualize_chain = contextualize_q_prompt | llm | StrOutputParser()

    def history_aware_retrieve(inputs):
        """Rewrite question if history exists, then retrieve documents."""
        if inputs.get("chat_history"):
            standalone_q = contextualize_chain.invoke(inputs)
        else:
            standalone_q = inputs["input"]
        return retriever.invoke(standalone_q)

    # ── Step 2: QA prompt — includes history for coherent multi-turn answers ─
    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", (
            "You are a careful RAG assistant. "
            "Answer using ONLY the context below. "
            "If the answer is not present, say: "
            '"I don\'t know based on the provided documents."\n\n'
            "Cite sources by their filename "
            "(e.g. 'According to report.pdf ...').\n\n"
            "Context:\n{context}"
        )),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])

    # ── Step 3: Answer chain (streaming-compatible) ───────────────────────────
    # This chain receives the full state dict {input, chat_history, context}
    # where context is still List[Document]. It formats context into a string
    # then streams the LLM response token by token.
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

    # ── Step 4: Wire retrieval + answer into one streaming LCEL chain ─────────
    # .assign(context=...) runs retrieval first. Output:
    #   {"input": str, "chat_history": list, "context": List[Document]}
    # .assign(answer=...) streams the LLM. Output adds:
    #   {"answer": "token1"}, {"answer": "token2"}, ...
    #
    # Streaming chunk order when .stream() is called:
    #   1. {"context": [doc1, doc2, ...]}  ← retrieval done (one chunk)
    #   2. {"answer": "The"}               ← LLM token 1
    #   3. {"answer": " report"}           ← LLM token 2  ...
    #
    # Keeping context as List[Document] (not formatting it here) means
    # source_list() can convert it for display after streaming completes.
    rag_chain = (
        RunnablePassthrough.assign(context=RunnableLambda(history_aware_retrieve))
        .assign(answer=answer_chain)
    )

    # ── Step 5: Auto history management ───────────────────────────────────────
    # RunnableWithMessageHistory loads history before each call and saves the
    # new (human, ai) pair after. You only need to pass session_id in config.
    return RunnableWithMessageHistory(
        rag_chain,
        _get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer",
    )
