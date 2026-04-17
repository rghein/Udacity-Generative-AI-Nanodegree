from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from typing import Dict, List, Optional

# RAGAS imports
try:
    from ragas import SingleTurnSample, EvaluationDataset, evaluate
    from ragas.metrics import (
        BleuScore,
        NonLLMContextPrecisionWithReference,
        ResponseRelevancy,
        Faithfulness,
        RougeScore,
    )
    RAGAS_AVAILABLE = True
except ImportError:
    RAGAS_AVAILABLE = False


def evaluate_response_quality(
    question: str,
    answer: str,
    contexts: List[str],
    reference: Optional[str] = None,
    enable_bleu: bool = False,
    enable_rouge: bool = False,
    enable_precision: bool = False,
) -> Dict[str, float]:
    """Evaluate response quality using RAGAS metrics.

    Required metrics:
    - ResponseRelevancy
    - Faithfulness

    Optional supported metrics:
    - BleuScore
    - RougeScore
    - NonLLMContextPrecisionWithReference (requires reference)
    """
    if not RAGAS_AVAILABLE:
        return {"error": "RAGAS not available"}

    if not contexts:
        return {"error": "No contexts provided for evaluation"}

    if not isinstance(question, str) or not question.strip():
        return {"error": "Question must be a non-empty string"}

    if not isinstance(answer, str) or not answer.strip():
        return {"error": "Answer must be a non-empty string"}

    if not isinstance(contexts, list) or not all(isinstance(c, str) for c in contexts):
        return {"error": "Contexts must be a non-empty list of strings"}

    if not contexts:
        return {"error": "No contexts provided for evaluation"}

    try:
        # TODO: Create evaluator LLM with model gpt-3.5-turbo
        evaluator_llm = LangchainLLMWrapper(
            ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
        )

        # TODO: Create evaluator_embeddings with model test-embedding-3-small
        evaluator_embeddings = LangchainEmbeddingsWrapper(
            OpenAIEmbeddings(model="text-embedding-3-small")
        )

        # TODO: Define an instance for each metric to evaluate

        # required metrics
        metrics = [
            ResponseRelevancy(),
            Faithfulness(),
        ]

        # optional metrics
        if enable_bleu:
            metrics.append(BleuScore())

        if enable_rouge:
            metrics.append(RougeScore())

        if enable_precision and reference:
            metrics.append(NonLLMContextPrecisionWithReference())
        
        # arguments for sample
        sample_kwargs = {
            "user_input": question,
            "response": answer,
            "retrieved_contexts": contexts,
        }

        # add reference if available
        if reference:
            sample_kwargs["reference"] = reference

        sample = SingleTurnSample(**sample_kwargs)
        dataset = EvaluationDataset(samples=[sample])

        # TODO: Evaluate the response using the metrics
        results = evaluate(
            dataset=dataset,
            metrics=metrics,
            llm=evaluator_llm,
            embeddings=evaluator_embeddings,
            raise_exceptions=False,
            show_progress=False,
        )

        # TODO: Return the evaluation results
        results_df = results.to_pandas()

        if results_df.empty:
            return {"error": "Evaluation returned no rows"}

        row = results_df.iloc[0].to_dict()

        scores = {
            key: float(value)
            for key, value in row.items()
            if isinstance(value, (int, float))
        }

        return scores

    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"}
    