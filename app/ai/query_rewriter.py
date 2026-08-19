from openai import AsyncOpenAI

from app.schemas.chat import Message
from app.observability.langfuse import langfuse


class QueryRewriter:
    def __init__(self, llm_client: AsyncOpenAI, model: str):
        self.llm_client = llm_client
        self.model = model

    async def rewrite(
        self,
        question: str,
        history: list[Message],
    ):
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

        with langfuse.start_as_current_observation(
            name="query-rewrite",
            as_type="generation",
            input={
                "question": question,
                "history": [msg.model_dump() for msg in history],
            },
            model=self.model,
        ) as observation:

            response = await self.llm_client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ],
                temperature=0,
            )

            standalone_question = (
                response.choices[0].message.content.strip()
            )

            observation.update(
                output=standalone_question
            )

            return standalone_question