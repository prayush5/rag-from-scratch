from openai import AsyncOpenAI
from app.core.config import settings
from llama_index.embeddings.jinaai import JinaEmbedding


client = AsyncOpenAI(
    api_key=settings.JINA_API_KEY,
    base_url="https://api.jina.ai/v1"
)

embedder = JinaEmbedding(
    api_key=settings.JINA_API_KEY,
    model=settings.EMBEDDING_MODEL,
    task="retrieval.query"
)

def embed_query(text: str):
    return embedder.get_text_embedding(text)

def embed_documents(documents: list[str]):
    return [
        embedder.get_text_embedding(document)
        for document in documents
    ]