import os
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from app.core.rate_limit import limiter

from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

from app.api.v1 import chat, ingest, documents
from app.services.rag_service import RAGService
from app.core.config import settings
from app.core.exceptions import AppException
from app.db.models import Base
from app.db.session import engine
from app.db.qdrant import init_collection
from app.ai.agent import compile_agent

os.makedirs("static", exist_ok=True)


def _to_psycopg_url(sqlalchemy_url: str) -> str:
    """psycopg (used by the LangGraph checkpointer) doesn't understand the
    '+asyncpg' driver suffix that SQLAlchemy uses — strip it down to a plain
    postgresql:// URL."""
    return sqlalchemy_url.replace("postgresql+asyncpg://", "postgresql://")


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Initializing RAG service...")
    app.state.rag_service = RAGService()
    init_collection()

    print("Initializing database tables...")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    print("Initializing agent checkpointer (Postgres)...")
    checkpointer_cm = AsyncPostgresSaver.from_conn_string(
        _to_psycopg_url(settings.DATABASE_URL)
    )
    checkpointer = await checkpointer_cm.__aenter__()
    await checkpointer.setup()
    app.state.agent = compile_agent(checkpointer)

    yield

    print("Shutting down agent checkpointer...")
    await checkpointer_cm.__aexit__(None, None, None)

    print("Shutting down database connection pool...")
    await engine.dispose()
    print("Shutting down...")


app = FastAPI(
    title="Document RAG",
    version="1.0.0",
    docs_url=None,
    redoc_url=None,
    openapi_url=None,
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8000"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# API Routers
app.include_router(chat.router, prefix="/api/v1", tags=["Chat"])
app.include_router(ingest.router, prefix="/api/v1", tags=["Ingest"])
app.include_router(documents.router, prefix="/api/v1", tags=["Documents"])

@app.get("/", include_in_schema=False)
async def serve_index():
    index_file = os.path.join("static", "index.html")
    if os.path.exists(index_file):
        return FileResponse(index_file)
    return JSONResponse({"message": "API is running. Visit /docs for OpenAPI documentation."})

@app.get("/health")
async def health_check():
    return {"status": "ok"}

@app.exception_handler(AppException)
async def app_exception_handler(request: Request, exc: AppException) -> JSONResponse:
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error_type": exc.__class__.__name__,
            "detail": exc.message,
            "path": request.url.path
        }
    )

app.mount("/static", StaticFiles(directory="static"), name="static")