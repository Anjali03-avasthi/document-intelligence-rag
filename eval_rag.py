import argparse
import json
import sys
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).parent
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

load_dotenv(PROJECT_ROOT / ".env")

from langchain_openai import ChatOpenAI
from chain import create_rag_chain, source_list
from database import is_index_current, load_vectorstore, rebuild_vectorstore
from loader import fingerprint_files, ingest_local_documents, list_supported_files
from config import ANSWER_MODEL


DATA_DIR = "data"
DEFAULT_EVAL_FILE = "eval_questions.json"


def _load_eval_questions(eval_file):
    with Path(eval_file).open("r", encoding="utf-8") as file:
        return json.load(file)


def _prepare_vectorstore(data_dir):
    files = list_supported_files(data_dir)
    if not files:
        raise RuntimeError(f"No PDF or CSV files found in {data_dir}/")

    document_fingerprint = fingerprint_files(files, data_dir)
    if is_index_current(document_fingerprint):
        return load_vectorstore(document_fingerprint), "loaded"

    documents, files = ingest_local_documents(data_dir)
    return (
        rebuild_vectorstore(
            documents=documents,
            document_fingerprint=document_fingerprint,
            file_count=len(files),
        ),
        "rebuilt",
    )


def _contains_expected_source(retrieved_sources, expected_source):
    expected_source = expected_source.lower()
    return any(
        expected_source in source["source"].lower()
        for source in retrieved_sources
    )


def _keyword_check(answer, retrieved_sources, expected_sources, expected_terms):
    """Original deterministic check: did expected sources and terms appear?"""
    missing_sources = [
        s for s in expected_sources
        if not _contains_expected_source(retrieved_sources, s)
    ]
    missing_terms = [
        t for t in expected_terms
        if t.lower() not in answer.lower()
    ]
    return {
        "passed": not missing_sources and not missing_terms,
        "missing_sources": missing_sources,
        "missing_terms": missing_terms,
    }


def _llm_judge(question: str, answer: str, context_docs: list, llm) -> dict:
    """Ask an LLM to score the answer on three criteria (1–5 each).

    Why LLM-as-judge?
    - Keyword checks are brittle: "tons" passes even if the number is wrong.
    - An LLM can assess meaning, coherence, and grounding — not just word presence.
    - This is also the foundation of agentic self-correction: agents that evaluate
      their own outputs and decide whether to retry or refine.

    Three criteria:
      faithfulness    — Is every claim in the answer grounded in the context?
                        Score 1 = hallucination, 5 = fully sourced.
      relevance       — Does the answer actually address the question asked?
                        Score 1 = off-topic, 5 = directly answers.
      context_quality — Did retrieval surface the right chunks?
                        Score 1 = wrong docs, 5 = exactly what was needed.
    """
    context_text = "\n---\n".join(
        f"Source: {doc.metadata.get('source', 'unknown')}\n{doc.page_content[:600]}"
        for doc in context_docs
    )

    judge_prompt = f"""You are an impartial RAG evaluation judge.

Question: {question}

Retrieved context:
{context_text}

System answer: {answer}

Rate the answer on these three criteria using integers 1–5:
- faithfulness: Is every claim backed by the context? (5=fully grounded, 1=hallucination)
- relevance: Does the answer directly address the question? (5=spot on, 1=off-topic)
- context_quality: Did retrieval surface the right information? (5=perfect, 1=irrelevant docs)

Reply ONLY with valid JSON — no markdown, no explanation outside the JSON:
{{"faithfulness": N, "relevance": N, "context_quality": N, "reason": "one sentence"}}"""

    try:
        response = llm.invoke(judge_prompt)
        return json.loads(response.content)
    except Exception as exc:
        return {
            "faithfulness": 0,
            "relevance": 0,
            "context_quality": 0,
            "reason": f"Parse error: {exc}",
        }


def run_eval(eval_file, data_dir, retrieval_k, use_judge=True):
    questions = _load_eval_questions(eval_file)
    vectorstore, index_status = _prepare_vectorstore(data_dir)
    retriever = vectorstore.as_retriever(search_kwargs={"k": retrieval_k})
    chain = create_rag_chain(retriever)

    # Reuse the same LLM for judging — cost-efficient since it's gpt-4o-mini.
    judge_llm = ChatOpenAI(model=ANSWER_MODEL, temperature=0) if use_judge else None

    print(f"Index status : {index_status}")
    print(f"Questions    : {len(questions)}")
    print(f"LLM judge    : {'enabled' if use_judge else 'disabled'}")
    print()

    passed_count = 0

    for index, item in enumerate(questions, start=1):
        question = item["question"]
        expected_sources = item.get("expected_sources", [])
        expected_terms = item.get("expected_terms", [])

        # Each question gets its own session so history doesn't bleed between them.
        result = chain.invoke(
            {"input": question},
            config={"configurable": {"session_id": f"eval_{index}"}},
        )

        answer = result["answer"]
        context_docs = result.get("context", [])
        retrieved_sources = source_list(context_docs)

        # ── Deterministic keyword check (original) ────────────────────────────
        keyword = _keyword_check(answer, retrieved_sources, expected_sources, expected_terms)
        if keyword["passed"]:
            passed_count += 1

        print(f"[{index}] {question}")
        print(f"Keyword check : {'PASS' if keyword['passed'] else 'FAIL'}")

        if keyword["missing_sources"]:
            print(f"  Missing sources : {', '.join(keyword['missing_sources'])}")
        if keyword["missing_terms"]:
            print(f"  Missing terms   : {', '.join(keyword['missing_terms'])}")

        # ── LLM-as-judge score ────────────────────────────────────────────────
        if use_judge and judge_llm:
            scores = _llm_judge(question, answer, context_docs, judge_llm)
            avg = (
                scores["faithfulness"] + scores["relevance"] + scores["context_quality"]
            ) / 3
            print(
                f"LLM judge     : "
                f"faithfulness={scores['faithfulness']}/5  "
                f"relevance={scores['relevance']}/5  "
                f"context={scores['context_quality']}/5  "
                f"avg={avg:.1f}/5"
            )
            print(f"  Judge reason  : {scores.get('reason', '')}")

        print("Retrieved sources:")
        for s in retrieved_sources:
            print(f"  [{s['id']}] {s['source']}")

        print("Answer:")
        print(result["answer"])
        print()

    print(f"Passed {passed_count}/{len(questions)} keyword checks")


def main():
    parser = argparse.ArgumentParser(description="Run RAG evaluation checks.")
    parser.add_argument("--eval-file", default=DEFAULT_EVAL_FILE)
    parser.add_argument("--data-dir", default=DATA_DIR)
    parser.add_argument("--k", type=int, default=5, help="Retrieved chunks per question")
    parser.add_argument(
        "--no-judge", action="store_true", help="Skip the LLM-as-judge scoring"
    )
    args = parser.parse_args()

    run_eval(args.eval_file, args.data_dir, args.k, use_judge=not args.no_judge)


if __name__ == "__main__":
    main()
