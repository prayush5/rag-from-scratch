from app.schemas.chunk import Chunk, EmbeddedChunk
from app.ai.client import client as ai_client

async def embed_chunks(chunks: list[Chunk]) -> list[EmbeddedChunk]:
    texts = [chunk.text for chunk in chunks]
    
    response = await ai_client.embeddings.create(
        model="jina-embeddings-v3",
        input=texts
    )

    embedded_chunks = []

    for chunk, embed in zip(chunks, response.data):
        embedded_chunks.append(
            EmbeddedChunk(
                text = chunk.text,
                source = chunk.source,
                filename = chunk.filename,
                chunk_index = chunk.chunk_index,
                embedding = embed.embedding
            )
        )

    return embedded_chunks
