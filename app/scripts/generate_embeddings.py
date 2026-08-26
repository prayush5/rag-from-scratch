import asyncio, json
from pathlib import Path
from app.ai.client import client as ai_client
from qdrant_client.models import PointStruct
from app.db.qdrant import client as qdrant_client

path = Path("data/questions.json")

with path.open("r", encoding="utf-8") as f:
    questions = json.load(f)

titles = [q["title"] for q in questions]
batch_size = 3

def chunk_list(items, batch_size):
    for i in range(0, len(items), batch_size):
        yield items[i: i+batch_size]

async def main():

    all_embeddings = []

    for batch in chunk_list(titles, batch_size):
        resp = await ai_client.embeddings.create(model="jina-embeddings-v3", input=batch)
        all_embeddings.extend([e.embedding for e in resp.data])
    
    points = []

    for question, embedding in zip(questions, all_embeddings):
        points.append(PointStruct(
            id = question["id"],
            vector = embedding,
            payload = {
                "title": question["title"],
                "language": question["language"],
                "category": question["category"],
                "difficulty": question["difficulty"],
            }
        ))

    qdrant_client.upsert(
        collection_name = "fastapi_questions",
        points = points
    )

    print(f"Inserted {len(points)} points into Qdrant.")
    

asyncio.run(main())
