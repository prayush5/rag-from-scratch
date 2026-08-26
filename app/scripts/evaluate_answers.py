import asyncio
import json

import scripts.llama_chat as rag

with open("data/evaluation/rag_eval.json", "r", encoding="utf-8") as f:
    evaluation_data = json.load(f)

#run evaluation
async def main():
    for index, item in enumerate(evaluation_data, start=1):
        question = item["question"]
        expected_sources = item["expected_sources"]
        reference_answer = item["reference_answer"]

        rag.conversation.clear()

        result = await rag.run_rag(question)

        print("\n" + "=" * 70)
        print(f"Evaluation #{index}")
        print("=" * 70)

        print(f"\nQuestion:")
        print(question)

        print(f"\nExpected Sources:")
        for source in expected_sources:
            print(f"- {source}")

        print(f"\nRetrieved Sources:")
        for source in result["sources"]:
            print(f"- {source}")

        print(f"\nStandalone Query:")
        print(result["standalone_query"])

        print(f"\nReference Answer:")
        print(reference_answer)

        print(f"\nGenerated Answer:")
        print(result["answer"])

        print(f"\nRetrieved Context:")
        print(result["context"])

if __name__ == "__main__":
    asyncio.run(main())
        
