from pydantic import BaseModel

class Document(BaseModel):
    text: str
    source: str
    filename: str