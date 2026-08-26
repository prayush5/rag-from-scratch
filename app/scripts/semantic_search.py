from app.ai.client import client
import asyncio
import numpy as np

documents = [
    "FastAPI is a modern Python framework.",
    "Cats enjoy sleeping all day.",
    "Django is a backend web framework.",
    "Pizza is delicious."
]

search_query = "Cats enjoy sleeping all day"

def cosine_similarity(v1,v2):
    return np.dot(v1,v2)/ (np.linalg.norm(v1) * np.linalg.norm(v2))

async def main():

    k = 3

    results = []

    docs_api_response, query_api_response = await asyncio.gather(
        client.embeddings.create(model="jina-embeddings-v3", input=documents),
        client.embeddings.create(model="jina-embeddings-v3", input=search_query)
    )

    query_vector = query_api_response.data[0].embedding

    for doc, doc_embedding in zip(documents, docs_api_response.data):
        score = cosine_similarity(doc_embedding.embedding, query_vector)
        
        results.append((doc,score))
    
    results.sort(key=lambda x: x[1], reverse=True)

    top_k_results = results[:k]

    for rank, (doc, score) in enumerate(top_k_results, start=1):
        print(f"#{rank} | Score: {score: .4f} | Document: {doc}")

asyncio.run(main())
