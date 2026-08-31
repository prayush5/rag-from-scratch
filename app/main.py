from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.api.v1 import chat, ingest, documents
from app.services.rag_service import RAGService
from app.core.exceptions import AppException
from app.db.models import Base
from app.db.session import engine

from app.db.qdrant import init_collection


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Initializing RAG service...")
    app.state.rag_service = RAGService()
    init_collection()

    print("Initializing database tables...")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    yield

    print("Shutting down database connection pool...")
    await engine.dispose()
    print("Shutting down...")


app = FastAPI(
    title="Document RAG",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8000"],
    allow_methods=["*"],
    allow_headers=["*"],
)


app.include_router(chat.router, prefix="/api/v1", tags=["Chat"])
app.include_router(ingest.router, prefix="/api/v1", tags=["Ingest"])
app.include_router(documents.router, prefix="/api/v1", tags=["Documents"])

@app.get("/health")
async def health_check():
    return {
        "status": "ok"
    }

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

app.mount("/", StaticFiles(directory="static", html=True), name="static")
