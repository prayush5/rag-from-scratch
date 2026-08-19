from dataclasses import dataclass, field

@dataclass
class IngestionStatus:
    status: str = "idle"
    documents_processed: int = 0
    nodes_created: int = 0
    error: str | None = None

ingestion_status = IngestionStatus()
