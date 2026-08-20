from fastapi import APIRouter, Depends, Request, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.schemas.chat import ChatRequest, ChatResponse
from app.services.rag_service import RAGService
from app.db.session import get_db
from app.db.repository.chat_repository import ChatRepository

router = APIRouter()

def get_rag_service(request: Request) -> RAGService:
    return request.app.state.rag_service

@router.post("/chat", response_model=ChatResponse)
async def chat_endpoint(
    payload: ChatRequest,
    rag_service: RAGService = Depends(get_rag_service),
    db: AsyncSession = Depends(get_db),
) -> ChatResponse:
    repo = ChatRepository(db)

    session_id = await repo.get_or_create_session(payload.session_id)
    history = await repo.get_history(session_id, limit=10)
    response = await rag_service.answer_question(question=payload.question, history=history)
    await repo.save_turn(session_id, payload.question, response.answer)
    response.session_id = session_id
    return response

@router.delete("/admin/sessions/cleanup", status_code=status.HTTP_200_OK)
async def cleanup_sessions(days: int = 30, db: AsyncSession = Depends(get_db)):
    repo = ChatRepository(db)
    deleted_count = await repo.cleanup_inactive_sessions(max_age_days=days)
    return {
        "status": "Success",
        "message": f"Cleaned up {deleted_count} inactive sessions."
    }
    