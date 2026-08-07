from app.ai.embedding import client as embedding_client
from app.db.qdrant import client as qdrant_client
from qdrant_client.models import Filter, FieldCondition, MatchValue

async def retrieve(query: str, limit: int = 10, score_threshold: float = 0.75, source: str | None = None):
    response = await embedding_client.embeddings.create(model="jina-embeddings-v3", input=query)
    query_embedding = response.data[0].embedding

    search_filter = None

    if source:
        search_filter = Filter(
            must=[
                FieldCondition(
                        key="source",
                        match=MatchValue(value=source)
                    )
                ]
            )
        
    search_results = qdrant_client.query_points(
        collection_name = "fastapi_documents",
        query = query_embedding,                                                                                        
        limit = limit,
        query_filter = search_filter
    )

    filtered = [
        point for point in search_results.points
        if point.score >= score_threshold
    ]

    return filtered