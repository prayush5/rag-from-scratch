from llama_index.core import Settings
from llama_index.embeddings.jinaai import JinaEmbedding
from llama_index.llms.groq import Groq
from app.core.config import settings

Settings.embed_model = JinaEmbedding(
    api_key=settings.JINA_API_KEY,
    model="jina-embeddings-v3",
    task="retrieval.passage",
)

Settings.llm = Groq(
    model=settings.LLM_MODEL,
    api_key=settings.GROQ_API_KEY,
)
