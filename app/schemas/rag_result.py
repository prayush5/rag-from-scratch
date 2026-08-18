from pydantic import BaseModel

class RAGResult(BaseModel):
    question: str
    standalone_query: str
    answer: str
    context: str
    sources: list[str]