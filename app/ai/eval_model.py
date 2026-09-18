from deepeval.models.base_model import DeepEvalBaseLLM
from langchain_groq import ChatGroq
from app.core.config import settings

class GroqEvalModel(DeepEvalBaseLLM):
    def __init__(self, model_name: str = None):
        self.model_name = model_name or settings.LLM_MODEL
        self.model = ChatGroq(
            model=self.model_name,
            api_key=settings.GROQ_API_KEY,
            temperature=0
        )

    def load_model(self):
        return self.model

    def generate(self, prompt: str):
        return self.load_model().invoke(prompt).content

    async def a_generate(self, prompt: str):
        res = await self.load_model().ainvoke(prompt)
        return res.content

    def get_model_name(self):
        return self.model_name

