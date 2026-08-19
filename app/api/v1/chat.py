from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks, Request
from app.schemas.chat import ChatRequest, ChatResponse
from app.schemas.ingest import IngestResponse
from app.services.rag_service import RAGService
from app.core.exceptions import RetrievalError, GenerationError

router = APIRouter()

def get_rag_service(request: Request) -> RAGService:
    return request.app.state.rag_service

@router.post("/chat", response_model=ChatResponse)
async def chat_endpoint(payload: ChatRequest, rag_service: RAGService = Depends(get_rag_service)):
    try:
        response = await rag_service.answer_question(question=payload.question, history=payload.history)
        return response
    except RetrievalError:
        raise HTTPException(
            status_code=503,
            detail="Document retrieval is currently unavailable."
        )
    except GenerationError:
        raise HTTPException(
            status_code=502,
            detail="Failed to generate response. Please try again."
        )