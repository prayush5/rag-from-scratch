import asyncio
from app.services.retriever import retrieve
from qdrant_client.models import Filter, FieldCondition, MatchValue

async def main():
    
    results = await retrieve("How do FastAPI dependencies work?", limit=10)

    for result in results:
        print("=" * 80)
        print(f"Score: {result.score:.3f}")
        print(f"Source: {result.payload['source']}")
        print(f"File: {result.payload['filename']}")
        print(f"Chunk: {result.payload['chunk_index']}")
        print()
        print(result.payload["text"])

asyncio.run(main())
