from pydantic import BaseModel, Field

class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    question: str = Field(..., description="User question")
    session_id: str | None = Field(default=None, description="Optional conversation session ID")

class ChatResponse(BaseModel):
    session_id: str | None = Field(default=None, description="Conversation session ID")
    question: str
    standalone_query: str
    answer: str
    sources: list[str]
    context: str