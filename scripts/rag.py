from app.services.generator import generate_answer
from app.services.retriever import retrieve
import asyncio

async def main():
    question = input("Ask anything: ")
    chunks = await retrieve(question, source = "sqlalchemy")
    print(f"Retrieved {len(chunks)} chunk(s)\n")

    for chunk in chunks:
        print("*" * 80)
        print(f"Score: {chunk.score:.3f}")
        print(f"{chunk.payload['source']}/{chunk.payload['filename']}")

    answer = await generate_answer(question, chunks)
    print("\n" + "=" * 80)
    print(answer)
    print("=" * 80)

if __name__ == "__main__":
    asyncio.run(main())
