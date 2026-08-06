from pathlib import Path
from app.schemas.document import Document

def load_documents(directory: str) -> list[Document]:
    documents = []

    for file in Path(directory).rglob("*.md"):
        text = file.read_text(encoding="utf-8")
        documents.append(
            Document(
                text=text,
                source=file.parent.name,
                filename=file.name
            )
        )

    return documents

if __name__ == "__main__":
    documents = load_documents("data")
    print(f"Loaded {len(documents)} documents.")
    
    for doc in documents:
        print("=" * 60)
        print(doc.filename)
        print(doc.source)
        print(doc.text[:500])