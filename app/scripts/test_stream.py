import asyncio
from llama_index.llms.groq import Groq
from app.core.config import settings

async def main():
    llm = Groq(model=settings.LLM_MODEL, api_key=settings.GROQ_API_KEY)
    import time
    start = time.time()
    first_chunk_time = None
    async for chunk in await llm.astream_complete("Write a 200 word story about a lighthouse."):
        if first_chunk_time is None:
            first_chunk_time = time.time() - start
            print(f"First chunk at: {first_chunk_time:.3f}s")
        print(chunk.delta, end="", flush=True)
    print(f"\nTotal: {time.time() - start:.3f}s")

asyncio.run(main())