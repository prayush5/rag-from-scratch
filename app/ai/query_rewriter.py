from openai import AsyncOpenAI
from app.schemas.chat import Message

class QueryRewriter:
    def __init__(self, llm_client: AsyncOpenAI, model: str):
        self.llm_client = llm_client
        self.model = model
    
    async def rewrite(self, question: str, history: list[Message]):
        if not history:
            return question
        
        history_str = "\n".join(
            f"{msg.role}: {msg.content}"
            for msg in history
        )

        prompt = f"""Given the conversation below, rewrite the user's latest question into a standalone question that can be understood without the conversation history.

Do not answer the question.

Conversation:
{history_str}

Latest question:
{question}

Standalone question:"""

        response = await self.llm_client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0
        )

        return response.choices[0].message.content.strip()
