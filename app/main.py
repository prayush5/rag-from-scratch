from contextlib import asynccontextmanager
from fastapi import FastAPI
from app.api.v1 import chat, ingest

@asynccontextmanager
async def lifespan(app: FastAPI):
    chat.get_rag_service()
    yield

app = FastAPI(
    title="Document RAG",
    version="1.0.0",
    lifespan=lifespan
)

app.include_router(chat.router, prefix="/api/v1", tags=["Chat"])
app.include_router(ingest.router, prefix="/api/v1", tags=["Ingest"])

@app.get("/health")
async def health_check():
    return {
        "status": "ok"
    }