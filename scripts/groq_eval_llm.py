import os
from deepeval.models.base_model import DeepEvalBaseLLM
from openai import AsyncOpenAI, OpenAI

class GroqLLMJudge(DeepEvalBaseLLM):
    def __init__(self, model_name: str = "openai/gpt-oss-120b"):
        self.model_name = model_name
        api_key = os.getenv("GROQ_API_KEY")
        base_url = "https://api.groq.com/openai/v1"

        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url
        )
        self.async_client = AsyncOpenAI(
            api_key=api_key,
            base_url=base_url
        )
    
    def load_model(self):
        return self.client

    def generate(self, prompt: str) -> str:
        res = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.0
        )

        return res.choices[0].message.content
    
    async def a_generate(self, prompt: str) -> str:
        res = await self.async_client.chat.completions.create(
            model=self.model_name,
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.0
        )

        return res.choices[0].message.content

    def get_model_name(self) -> str:
        return "openai/gpt-oss-120b"
