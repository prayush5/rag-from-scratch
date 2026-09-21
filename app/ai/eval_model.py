import os
from pydantic import BaseModel
from deepeval.models.base_model import DeepEvalBaseLLM
from langchain_groq import ChatGroq
from langchain_ollama import ChatOllama
from app.core.config import settings


class UniversalEvalModel(DeepEvalBaseLLM):
    """Judge model used only for DeepEval scoring — independent of the
    agent's own model, which always stays on Groq to match production."""

    def __init__(self, model_name: str = None):
        self.provider = os.getenv("EVAL_LLM_PROVIDER", "groq").lower()
        self.model_name = model_name or os.getenv("EVAL_LLM_MODEL", settings.LLM_MODEL)

        if self.provider == "ollama":
            self.model = ChatOllama(
                model=self.model_name,
                base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
                temperature=0,
                timeout=120
            )
        else:
            self.model = ChatGroq(
                model=self.model_name,
                api_key=settings.GROQ_API_KEY,
                temperature=0,
            )

    def load_model(self):
        return self.model

    def generate(self, prompt: str) -> str:
        return self.load_model().invoke(prompt).content

    async def a_generate(self, prompt: str) -> str:
        res = await self.load_model().ainvoke(prompt)
        return res.content

    def generate_with_schema(self, prompt: str, schema: BaseModel) -> BaseModel:
        structured_llm = self.load_model().with_structured_output(schema)
        return structured_llm.invoke(prompt)

    async def a_generate_with_schema(self, prompt: str, schema: BaseModel) -> BaseModel:
        structured_llm = self.load_model().with_structured_output(schema)
        return await structured_llm.ainvoke(prompt)

    def get_model_name(self) -> str:
        return f"{self.provider}:{self.model_name}"