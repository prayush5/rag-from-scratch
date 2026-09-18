import json
import asyncio
import sys
from deepeval.metrics import FaithfulnessMetric, AnswerRelevancyMetric
from deepeval.test_case import LLMTestCase
from langchain_core.messages import HumanMessage
from pathlib import Path
from app.ai.eval_model import GroqEvalModel

from app.ai.agent import compile_agent

groq_judge = GroqEvalModel()

SCORE_THRESHOLD = 0.7
DATASET_PATH = Path(__file__).resolve().parent.parent / "tests" / "data" / "golden_dataset.json"

async def evaluate_agent():
    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        dataset = json.load(f)
    
    agent = compile_agent(checkpointer=None)

    faithfulness_metric = FaithfulnessMetric(threshold=SCORE_THRESHOLD, model=groq_judge)
    relevancy_metric = AnswerRelevancyMetric(threshold=SCORE_THRESHOLD, model=groq_judge)

    failed_cases = 0

    print(f"Starting deepeval regression testing ({len(dataset)} test cases)")

    for idx, item in enumerate(dataset, 1):
        question = item["input"]
        print(f"\n[Case {idx}/{len(dataset)}] Question: {question}")

        inputs = {
            "question": question,
            "messages": [HumanMessage(content=question)]
        }

        res = await agent.ainvoke(inputs)

        actual_output = res["messages"][-1].content
        retrieved_context = res.get("tool_output", item["context"])

        test_case = LLMTestCase(
            input=question,
            actual_output=actual_output,
            retrieval_context=[retrieved_context] if isinstance(retrieved_context, str) else retrieved_context,
            expected_output=item.get("expected_output")
        )

        faithfulness_metric.measure(test_case)
        relevancy_metric.measure(test_case)
        
        f_score = faithfulness_metric.score
        r_score = relevancy_metric.score

        print(f"   - Faithfulness: {f_score:.2f} (Passed: {faithfulness_metric.is_successful()})")
        print(f"   - Relevancy:    {r_score:.2f} (Passed: {relevancy_metric.is_successful()})")

        if not faithfulness_metric.is_successful() or not relevancy_metric.is_successful():
            failed_cases += 1
            print(f"   FAILED case {idx}")

    print(f"\n================ Evaluation Summary ================")
    print(f"Total: {len(dataset)} | Passed: {len(dataset) - failed_cases} | Failed: {failed_cases}")

    if failed_cases > 0:
        print("Regression test failed! Minimum threshold requirement not met.")
        sys.exit(1)
    
    print("All evaluation metrics passed successfully!")

if __name__ == "__main__":
    asyncio.run(evaluate_agent())