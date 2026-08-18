from pydantic import BaseModel, Field

class Message(BaseModel):
    role: str = Field(..., description="Role of the speaker ('user' or 'assistant')")
    content: str = Field(..., description="Content of the message")

class ChatRequest(BaseModel):
    question: str = Field(..., min_length=10, description="The latest user question")
    history: list[Message] = Field(default_factory=list, description="Previous conversation history")

class ChatResponse(BaseModel):
    question: str
    standalone_query: str
    answer: str
    sources: list[str]
    context: str