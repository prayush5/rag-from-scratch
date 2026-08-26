from qdrant_client import AsyncQdrantClient, QdrantClient
from qdrant_client.models import Distance, VectorParams
from app.core.config import settings

COLLECTION_NAME = "fastapi_documents"

client = QdrantClient(
    url=settings.QDRANT_URL
)

async_client = AsyncQdrantClient(
    url=settings.QDRANT_URL
)

def create_collection():
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(
            size=1024,
            distance=Distance.COSINE
        )
    )

def init_collection():
    if not client.collection_exists(COLLECTION_NAME):
        create_collection()

def recreate_collection():
    if client.collection_exists(COLLECTION_NAME):
        client.delete_collection(COLLECTION_NAME)
    
    create_collection()

