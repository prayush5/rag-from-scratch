import httpx

from app.core.config import settings

JINA_RERANK_URL = "https://api.jina.ai/v1/rerank"
RERANK_MODEL = "jina-reranker-v2-base-multilingual"

async def rerank(query: str, documents: list[str], top_n: int = 3):
    payload = {
        "model": RERANK_MODEL,
        "query": query,
        "documents": documents,
        "top_n": top_n
    }

    headers = {
        "Authorization": f"Bearer {settings.JINA_API_KEY}",
        "Content-Type": "application/json"
    }

    async with httpx.AsyncClient() as client:
        response = await client.post(
            JINA_RERANK_URL,
            json=payload,
            headers=headers
        )
    response.raise_for_status()

    return response.json()
        
        