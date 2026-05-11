from pathlib import Path

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


def _source_name(document):
    source = document.metadata.get("source", "Unknown source")
    name = Path(source).name

    if "page" in document.metadata:
        return f"{name}, page {document.metadata['page'] + 1}"

    if "row" in document.metadata:
        return f"{name}, row {document.metadata['row']}"

    return name


def _format_documents(documents):
    formatted = []
    for index, document in enumerate(documents, start=1):
        source = _source_name(document)
        formatted.append(
            f"[{index}] Source: {source}\n{document.page_content}"
        )
    return "\n\n".join(formatted)


def _source_list(documents):
    sources = []
    for index, document in enumerate(documents, start=1):
        sources.append(
            {
                "id": index,
                "source": _source_name(document),
                "preview": document.page_content[:400],
            }
        )
    return sources


def create_rag_chain(retriever):
    template = """You are a careful RAG assistant.
Answer the question using only the context below.
If the answer is not present in the context, say: "I don't know based on the provided documents."

Use citations like [1], [2] when you use information from a source.

Context:
{context}

Question:
{question}
"""

    prompt = ChatPromptTemplate.from_template(template)
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
    answer_chain = prompt | llm | StrOutputParser()

    def ask(question):
        documents = retriever.invoke(question)
        context = _format_documents(documents)
        answer = answer_chain.invoke({"context": context, "question": question})

        return {
            "answer": answer,
            "sources": _source_list(documents),
        }

    return ask
