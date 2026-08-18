import asyncio
import json

from openai import AsyncOpenAI

import scripts.llama_chat as rag
from app.core.config import settings
from app.schemas.faithfulness_eval import FaithfulnessEvaluation


client = AsyncOpenAI(
    api_key=settings.GROQ_API_KEY,
    base_url="https://api.groq.com/openai/v1"
)

MODEL = "openai/gpt-oss-120b"


# Load evaluation dataset
with open(
    "data/evaluation/rag_eval.json",
    "r",
    encoding="utf-8"
) as f:
    evaluation_data = json.load(f)


# Faithfulness evaluator
async def evaluate_faithfulness(
    context: str,
    answer: str
) -> FaithfulnessEvaluation:

    prompt = f"""
You are a strict evaluator for a Retrieval-Augmented Generation (RAG) system.

Determine whether every factual claim in the answer is supported
by the provided context.

Rules:

1. Use ONLY the provided context.
2. Do NOT use outside knowledge.
3. Check each factual claim in the answer individually.
4. If even one factual claim is not supported by the context,
   faithful must be false.
5. Minor wording differences are acceptable.
6. Do not judge whether the answer is generally true.
   Judge whether it is supported by THIS context.
7. Return ONLY a JSON object.
8. Do NOT provide explanations outside the JSON object.

Context:
{context}

Answer:
{answer}

Your response MUST have exactly these fields:

{{
    "faithful": true,
    "unsupported_claims": [],
    "reason": "..."
}}
"""

    response = await client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0,
    )

    content = response.choices[0].message.content.strip()

    print("\nRAW JUDGE RESPONSE:")
    print(content)

    # Try to extract JSON object from the response
    start = content.find("{")
    end = content.rfind("}")

    if start == -1 or end == -1:
        raise ValueError(
            "Judge did not return a JSON object.\n"
            f"Raw response:\n{content}"
        )

    json_content = content[start:end + 1]

    data = json.loads(json_content)

    return FaithfulnessEvaluation.model_validate(data)

# Run evaluation
async def main():

    faithful_scores = []

    for index, item in enumerate(   
        evaluation_data,
        start=1
    ):
        question = item["question"]

        # Keep evaluation questions independent
        rag.conversation.clear()

        result = await rag.run_rag(question)

        evaluation = await evaluate_faithfulness(
            context=result["context"],
            answer=result["answer"]
        )

        faithful = evaluation.faithful

        faithful_scores.append(int(faithful))

        print("\n" + "=" * 70)
        print(f"Evaluation #{index}")
        print("=" * 70)

        print("\nQuestion:")
        print(question)

        print("\nGenerated Answer:")
        print(result["answer"])

        print("\nFaithful:")
        print(faithful)

        print("\nUnsupported Claims:")

        if evaluation.unsupported_claims:
            for claim in evaluation.unsupported_claims:
                print(f"- {claim}")
        else:
            print("- None")

        print("\nReason:")
        print(evaluation.reason)

    # Final score
    faithfulness_score = (
        sum(faithful_scores)
        / len(faithful_scores)
    )

    print("\n")
    print("=" * 70)
    print("FINAL FAITHFULNESS EVALUATION")
    print("=" * 70)

    print(f"Questions:       {len(faithful_scores)}")
    print(
        f"Faithfulness:    "
        f"{faithfulness_score:.3f}"
    )


if __name__ == "__main__":
    asyncio.run(main())