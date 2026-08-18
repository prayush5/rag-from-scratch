import asyncio
from app.ai.reranker import rerank

async def main():
    query = "How does UploadFile handles large files?"
    documents = [
        """
        FastAPI middleware allows processing requests
        and responses before and after path operations.
        """,

        """
        UploadFile uses SpooledTemporaryFile to efficiently
        handle large files without consuming all system memory.
        It also provides asynchronous read, write, and close methods.
        """,

        """
        FastAPI Dependency Injection allows dependencies to
        be declared and automatically resolved.
        """
    ]

    results = await rerank(query=query, documents=documents, top_n= 3)

    print("\nReranked results:")

    for result in results["results"]:
        print(
            f"\nIndex: {result['index']}"
            f"\nScore: {result['relevance_score']}"
            f"\nDocument: {documents[result['index']]}"
        )

if __name__ == "__main__":
    asyncio.run(main())
