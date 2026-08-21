import json
from fastapi import APIRouter, Depends, Request, status
from fastapi.responses import StreamingResponse
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

@router.post("/chat/stream")
async def chat_stream(
    payload: ChatRequest,
    rag_service: RAGService = Depends(get_rag_service),
    db: AsyncSession = Depends(get_db),
):
    repo = ChatRepository(db)

    session_id = await repo.get_or_create_session(payload.session_id)
    history = await repo.get_history(session_id, limit=10)

    async def event_generator():
        accumulated_answer = []
        try:
            async for event_type, data in rag_service.stream_answer_question(payload.question, history):
                if event_type == "token":
                    accumulated_answer.append(data)
                    yield f"data: {json.dumps({'type': 'token', 'content': data})}\n\n"
                
                elif event_type == "metadata":
                    yield f"data: {json.dumps({'type': 'metadata', 'session_id': session_id, 'sources': data['sources']})}\n\n"

        finally:
            if accumulated_answer:
                full_response = "".join(accumulated_answer)
                await repo.save_turn(session_id, payload.question, full_response)

        yield f"data: {json.dumps({'type': 'done'})}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")