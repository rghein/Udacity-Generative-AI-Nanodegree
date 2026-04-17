import os
import json
import argparse

from ragas_evaluator import evaluate_response_quality
from rag_client import initialize_rag_system, retrieve_documents, format_context
from llm_client import generate_response

def load_questions(test_file: str):
    """Load questions from JSON or plain text file"""

    if test_file.endswith(".json"):
        with open(test_file, "r", encoding="utf-8") as f:
            data = json.load(f)
            questions = []
            for item in data:
                if isinstance(item, dict) and "question" in item and item["question"].strip():
                    questions.append(item["question"].strip())
            if not questions:
                raise ValueError("No valid questions found in JSON file")
            return questions

    elif test_file.endswith(".txt"):
        with open(test_file, "r", encoding="utf-8") as f:
            return [line.strip() for line in f if line.strip()]

    else:
        raise ValueError("Unsupported file format. Use .json or .txt")

def run_dataset_evaluation(test_file, collection, openai_key, model="gpt-3.5-turbo", n_docs=3):
    questions = load_questions(test_file)

    results = []

    for question in questions:
        docs = retrieve_documents(collection, question, n_results=n_docs)

        context = ""
        contexts_list = []

        if docs and docs.get("documents"):
            context = format_context(docs["documents"][0], docs["metadatas"][0])
            contexts_list = docs["documents"][0]

        answer = generate_response(openai_key, question, context, [], model)

        scores = evaluate_response_quality(question, answer, contexts_list)

        print(f"\nQuestion: {question}")
        print(f"Answer: {answer}")
        print(f"Scores: {scores}")

        results.append({
            "question": question,
            "answer": answer,
            "scores": scores
        })

    metrics = {}
    for result in results:
        for key, value in result["scores"].items():
            if isinstance(value, (int, float)):
                metrics.setdefault(key, []).append(value)

    print("\n=== Aggregate Metrics ===")
    for key, values in metrics.items():
        print(f"{key}: {sum(values) / len(values):.3f}")

def main():
    parser = argparse.ArgumentParser(description="Evaluate NASA RAG system on a dataset of questions")
    parser.add_argument("--test-file", required=True, help="Path to .json or .txt file of test questions")
    parser.add_argument("--openai-key", default=os.getenv("OPENAI_API_KEY", ""), help="OpenAI API key")
    parser.add_argument("--chroma-dir", default="./chroma_db_openai", help="ChromaDB persist directory")
    parser.add_argument("--collection-name", default="nasa_space_missions_text", help="Collection name")
    parser.add_argument("--model", default="gpt-3.5-turbo", help="OpenAI model for generation")
    parser.add_argument("--n-docs", type=int, default=3, help="Number of documents to retrieve")
    args = parser.parse_args()

    if not args.openai_key:
        raise ValueError("Missing OpenAI API key")

    os.environ["OPENAI_API_KEY"] = args.openai_key
    os.environ["CHROMA_OPENAI_API_KEY"] = args.openai_key

    collection = initialize_rag_system(args.chroma_dir, args.collection_name)

    run_dataset_evaluation(
        test_file=args.test_file,
        collection=collection,
        openai_key=args.openai_key,
        model=args.model,
        n_docs=args.n_docs
    )

if __name__ == "__main__":
    main()
