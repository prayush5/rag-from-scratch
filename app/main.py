from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from app.api.v1 import chat, ingest
from app.services.rag_service import RAGService
from app.core.exceptions import AppException
from fastapi.responses import JSONResponse

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Initializing RAG service...")
    app.state.rag_service = RAGService()
    yield
    print("Shutting down...")

app = FastAPI(
    title="Document RAG",
    version="1.0.0",
    lifespan=lifespan
)

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

app.include_router(chat.router, prefix="/api/v1", tags=["Chat"])
app.include_router(ingest.router, prefix="/api/v1", tags=["Ingest"])

@app.get("/health")
async def health_check():
    return {
        "status": "ok"
    }