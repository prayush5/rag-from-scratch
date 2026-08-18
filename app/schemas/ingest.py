from pydantic import BaseModel

class IngestResponse(BaseModel):
    status: str
    documents_processed: int
    nodes_created: int