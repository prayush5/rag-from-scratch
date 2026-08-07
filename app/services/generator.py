from app.ai.llm import client as ai_client
from app.core.config import settings
from qdrant_client.models import ScoredPoint

async def generate_answer(question: str, chunks: list[ScoredPoint]) -> str:
    context = build_context(chunks)
    prompt = build_prompt(question, context)
    response = await ai_client.chat.completions.create(
        model = settings.LLM_MODEL,
        messages = [
            {
                "role": "user",
                "content": prompt
            }
        ],
    )

    return response.choices[0].message.content
    

def build_context(chunks: list[ScoredPoint]) -> str:
    sections = []

    for chunk in chunks:
        payload = chunk.payload

        sections.append(
            f"""Source: {payload['source']}/{payload['filename']}
        Chunk: {payload['chunk_index']}

        {payload['text']}"""
        )

    return "\n\n========================================\n\n".join(sections)

def build_prompt(question: str, context: str) -> str:
    return f"""
You are an AI assistant that answers questions using the provided documentation.

Instructions:
- Use ONLY the provided context.
- If the provided context does not contain enough information to answer the question, explicitly say so instead of making assumptions.
- Do not make up information.
- Answer clearly, concisely, and professionally.
- Cite the relevant source(s) from the context when possible.

Context:
{context}

Question:
{question}

Answer:"""


