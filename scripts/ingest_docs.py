from app.services.embedder import embed_chunks
from app.services.loader import load_documents
from app.services.chunker import chunk_document
from app.schemas.chunk import EmbeddedChunk
from app.db.qdrant import client as qdrant_client, recreate_collection
from qdrant_client.models import PointStruct
from uuid import uuid4
import asyncio

async def main():

    recreate_collection()

    documents = load_documents("data")

    all_chunks = []

    for document in documents:
        all_chunks.extend(chunk_document(document))

    embedded_chunks = await embed_chunks(all_chunks)
    points = []

    for embedding in embedded_chunks:
        points.append(PointStruct(
            id = uuid4(),
            vector = embedding.embedding,
            payload = {
                "text": embedding.text,
                "source": embedding.source,
                "filename": embedding.filename,
                "chunk_index": embedding.chunk_index,
                "length": len(embedding.text)
            }
        ))

    qdrant_client.upsert(
        collection_name = "fastapi_documents",
        points = points
    )

    print(f"Inserted {len(points)} points into Qdrant.")

if __name__ == "__main__":
    asyncio.run(main())