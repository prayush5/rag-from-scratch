from app.schemas.document import Document
from app.schemas.chunk import Chunk
from app.services.loader import load_documents

def chunk_document(document: Document, chunk_size: int = 500, overlap: int = 100) -> list[Chunk]:
    text = document.text
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(
            Chunk(
                text=text[start:end],
                source=document.source,
                filename=document.filename,
                chunk_index=len(chunks)
            )
        )
        start += chunk_size - overlap
    return chunks

if __name__ == "__main__":
    documents = load_documents("data")

    first_doc = documents[0]

    chunks = chunk_document(first_doc)

    print(f"{first_doc.filename} produced {len(chunks)} chunks\n")

    for chunk in chunks:
        print("=" * 60)
        print(f"Chunk {chunk.chunk_index}")
        print(chunk.text[:500])