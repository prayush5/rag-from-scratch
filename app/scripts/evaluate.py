import asyncio
from dotenv import load_dotenv

load_dotenv()

from deepeval import evaluate
from deepeval.metrics import FaithfulnessMetric, AnswerRelevancyMetric, ContextualPrecisionMetric
from deepeval.test_case import LLMTestCase

from app.services.rag_service import RAGService
from app.scripts.groq_eval_llm import GroqLLMJudge

TEST_DATASET = [
    {
        "input": "How do I configure dependency injection in FastAPI?",
        "expected_output": "Use Depends() from fastapi and inject it in the endpoint function parameters.",
    },
    {
        "input": "What is the chunking strategy used for document ingestion?",
        "expected_output": "Parent-child chunking where small child chunks are indexed and parent chunks are returned.",
    }
]

def extract_node_text(node) -> str:
    if isinstance(node, str):
        return node
    target = getattr(node, "node", node)
    if hasattr(target, "get_content"):
        return target.get_content()
    if hasattr(target, "text"):
        return target.text
    return str(target)

async def main():
    rag_service = RAGService()
    judge = GroqLLMJudge()

    for idx, sample in enumerate(TEST_DATASET, start=1):
        print(f"\n--- Running Evaluation [{idx}/{len(TEST_DATASET)}]: {sample['input']} ---")

        result = await rag_service.answer_question(sample['input'], history=[])
        
        # Safely pull attributes from Pydantic / dataclass / dict
        if isinstance(result, dict):
            actual_answer = result.get("answer", str(result))
            raw_context = result.get("context_nodes", result.get("context", []))
        else:
            actual_answer = getattr(result, "answer", str(result))
            raw_context = getattr(result, "context_nodes", getattr(result, "context", []))

        # Format retrieval context into a list of strings
        if isinstance(raw_context, str):
            retrieval_context = [raw_context]
        else:
            retrieval_context = [extract_node_text(node) for node in raw_context]

        test_case = LLMTestCase(
            input=sample['input'],
            actual_output=actual_answer,
            retrieval_context=retrieval_context,
            expected_output=sample['expected_output']
        )

        faithfulness = FaithfulnessMetric(threshold=0.7, model=judge)
        relevancy = AnswerRelevancyMetric(threshold=0.7, model=judge)
        precision = ContextualPrecisionMetric(threshold=0.7, model=judge)

        evaluate(test_cases=[test_case], metrics=[faithfulness, relevancy, precision])

        if idx < len(TEST_DATASET):
            print("\nPausing 15 seconds to reset TPM rate limits...")
            await asyncio.sleep(15)

if __name__ == "__main__":
    asyncio.run(main())