from pydantic import BaseModel

class Chunk(BaseModel):
    text: str
    source: str
    filename: str
    chunk_index: int

class EmbeddedChunk(BaseModel):
    text: str
    source: str
    filename: str
    chunk_index: int
    embedding: list[float]