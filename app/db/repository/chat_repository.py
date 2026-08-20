import uuid
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, delete, update
from app.db.models import ChatSessionModel, ChatMessageModel
from app.schemas.chat import Message
from datetime import datetime, timezone, timedelta

class ChatRepository:
    def __init__(self, db_session: AsyncSession):
        self.db = db_session

    async def get_or_create_session(self, session_id: str | None = None) -> str:
        if session_id:
            stmnt = select(ChatSessionModel).where(ChatSessionModel.id == session_id)
            result = await self.db.execute(stmnt)
            existing_session = result.scalar_one_or_none()
            if existing_session:
                return existing_session.id
        
        new_session = ChatSessionModel(id=session_id or str(uuid.uuid4()))
        self.db.add(new_session)
        await self.db.commit()
        return new_session.id
    

    async def get_history(self, session_id: str, limit: int = 10) -> list[Message]:
        stmnt = (
            select(ChatMessageModel)
            .where(ChatMessageModel.session_id == session_id)
            .order_by(ChatMessageModel.created_at.desc())
            .limit(limit)
        )
        result = await self.db.execute(stmnt)
        records = reversed(result.scalars().all())
        return [Message(role=m.role, content=m.content) for m in records]


    async def save_turn(self, session_id: str, user_question: str, assistant_answer: str) -> None:
        user_msg = ChatMessageModel(session_id=session_id, role="user", content=user_question)
        assistant_msg = ChatMessageModel(session_id=session_id, role="assistant", content=assistant_answer)

        self.db.add_all([user_msg, assistant_msg])
        stmnt = (
            update(ChatSessionModel)
            .where(ChatSessionModel.id == session_id)
            .values(updated_at=datetime.now(timezone.utc))
        )

        await self.db.execute(stmnt)
        await self.db.commit()


    async def cleanup_inactive_sessions(self, max_age_days: int = 30) -> int:
        cutoff_date = datetime.now(timezone.utc) - timedelta(days=max_age_days)

        stmt = (
            delete(ChatSessionModel)
            .where(ChatSessionModel.updated_at < cutoff_date)
        )
        result = await self.db.execute(stmt)
        await self.db.commit()
        return result.rowcount 
    