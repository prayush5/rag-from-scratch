from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from app.schemas.chat import ChatRequest, ChatResponse
from app.schemas.ingest import IngestResponse
from app.services.rag_service import RAGService

router = APIRouter()

_rag_service : RAGService | None = None

def get_rag_service() -> RAGService:
    global _rag_service
    if _rag_service is None:
        _rag_service = RAGService()
    return _rag_service

@router.post("/chat", response_model=ChatResponse)
async def chat_endpoint(payload: ChatRequest, rag_service: RAGService = Depends(get_rag_service)):
    try:
        response = await rag_service.answer_question(question=payload.question, history=payload.history)
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
