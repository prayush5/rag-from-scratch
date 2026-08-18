from fastapi import APIRouter

from app.schemas.ingest import IngestResponse
from scripts.ingest_docs import run_ingestion

router = APIRouter()

@router.post("", response_model=IngestResponse)
def ingest_documents():
    result = run_ingestion()

    return IngestResponse(
        status="success",
        documents_processed=result["documents_processed"],
        nodes_created=result["nodes_created"]
    )