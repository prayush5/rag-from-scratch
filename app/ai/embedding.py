from openai import AsyncOpenAI
from app.core.config import settings

client = AsyncOpenAI(
    api_key=settings.JINA_API_KEY,
    base_url="https://api.jina.ai/v1"
)