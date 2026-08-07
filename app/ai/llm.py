from openai import AsyncOpenAI
from app.core.config import settings

client = AsyncOpenAI(
    api_key=settings.GROQ_API_KEY,
    base_url="https://api.groq.com/openai/v1",
)