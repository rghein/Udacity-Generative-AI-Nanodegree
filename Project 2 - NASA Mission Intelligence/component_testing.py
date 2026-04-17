import os
import subprocess

from llm_client import generate_response
from rag_client import discover_chroma_backends
from ragas_evaluator import evaluate_response_quality

def test_llm():
    print("\n=== LLM TEST ===")
    api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        print("ERROR: OPENAI_API_KEY not set")
        return

    response = generate_response(
        api_key,
        "What was Apollo 11?",
        "",
        []
    )

    print("Response:", response[:200], "...")

def test_rag_backend():
    print("\n=== RAG BACKEND TEST ===")
    backends = discover_chroma_backends()
    print("Backends:", backends)

def test_embedding_pipeline():
    print("\n=== EMBEDDING PIPELINE TEST ===")

    api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        print("ERROR: OPENAI_API_KEY not set")
        return

    # Run CLI command
    result = subprocess.run(
        [
            "python",
            "embedding_pipeline.py",
            "--openai-key",
            api_key,
            "--stats-only"
        ],
        capture_output=True,
        text=True
    )

    if result.stdout:
        print(result.stdout)
    if result.stderr:
        print(result.stderr)

def test_ragas():
    print("\n=== RAGAS TEST ===")

    scores = evaluate_response_quality(
        "What was Apollo 11?",
        "Apollo 11 was the first mission to land humans on the Moon.",
        ["Apollo 11 was the first mission to land humans on the Moon in 1969."]
    )

    print("Scores:", scores)

def main():
    print("\n======Component Testing======")

    test_llm()
    test_rag_backend()
    test_embedding_pipeline()
    test_ragas()
    print()

if __name__ == "__main__":
    main()
