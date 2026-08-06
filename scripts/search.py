from app.ai.client import client as ai_client
from app.db.qdrant import client as qdrant_client
import asyncio
from qdrant_client.models import Filter, FieldCondition, MatchValue

search_query = "How do FastAPI dependencies work?"

async def main():
    response = await ai_client.embeddings.create(model="jina-embeddings-v3", input=search_query)
    query_embedding = response.data[0].embedding

    search_results = qdrant_client.query_points(
        collection_name = "fastapi_documents",
        query = query_embedding,                                                                                        
        limit=3
    )

    for result in search_results.points:
        print("=" * 80)
        print(f"Score: {result.score:.3f}")
        print(f"Source: {result.payload['source']}")
        print(f"File: {result.payload['filename']}")
        print(f"Chunk: {result.payload['chunk_index']}")
        print()
        print(result.payload["text"])

asyncio.run(main())