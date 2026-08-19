from fastapi import APIRouter, BackgroundTasks, HTTPException

from app.schemas.ingest import IngestResponse
from scripts.ingest_docs import run_ingestion
from app.services.ingestion_status import ingestion_status
from app.core.exceptions import IngestionError

router = APIRouter()

def run_ingestion_background():
    ingestion_status.status = "running"

    try:
        result = run_ingestion()
        ingestion_status.documents_processed = result["documents_processed"]
        ingestion_status.nodes_created = result["nodes_created"]
        ingestion_status.status = "completed"

    except Exception:
        ingestion_status.status = "failed"
        ingestion_status.error = "Documemnt ingestion failed"


@router.post("", response_model=IngestResponse)
def ingest_documents(background_tasks: BackgroundTasks):

    if ingestion_status.status in {"running", "started"}:
        raise HTTPException(
            status_code=409,
            detail="Ingestion already in progress",
        )

    ingestion_status.status = "started"
    ingestion_status.error = None
    ingestion_status.documents_processed = 0
    ingestion_status.nodes_created = 0

    background_tasks.add_task(run_ingestion_background)

    return IngestResponse(
        status="started",
        message="Ingestion started in background",
        documents_processed=0,
        nodes_created=0
    )

@router.get("/status", response_model=IngestResponse)
def ingestion_status_endpoint():
    return IngestResponse(
        status=ingestion_status.status,
        message=(
            "Ingestion completed"
            if ingestion_status.status == "completed"
            else "Ingestion failed"
            if ingestion_status.status == "failed"
            else "Ingestion in progress"
        ),
        documents_processed=ingestion_status.documents_processed,
        nodes_created=ingestion_status.nodes_created,
    )
