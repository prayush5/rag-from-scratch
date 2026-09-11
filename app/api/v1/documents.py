import os
import shutil
from fastapi import APIRouter, File, HTTPException, UploadFile, status, BackgroundTasks, Depends, Request
from app.scripts.ingest_docs import run_ingestion
from app.core.security import verify_admin
from app.core.rate_limit import limiter

router = APIRouter(prefix="/documents", tags=["Documents"])

MAX_FILE_SIZE = 5 * 1024 * 1024
ALLOWED_EXTS = {".md", ".pdf", ".docx"}
DATA_DIR = "./data"

@router.post("/upload", summary="Upload new documents for ingestion", status_code=status.HTTP_202_ACCEPTED)
@limiter.limit("3/minute")
async def upload_document(request: Request, background_tasks: BackgroundTasks, file: UploadFile = File(...), _: None = Depends(verify_admin)):
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in ALLOWED_EXTS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported extension '{ext}'. Only {', '.join(ALLOWED_EXTS)} are allowed"
        )
    
    contents = await file.read()
    if len(contents) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File too large. Max size is {(MAX_FILE_SIZE/1024)/1024}MB"
        )
    
    os.makedirs(DATA_DIR, exist_ok=True)
    file_path = os.path.join(DATA_DIR, file.filename)

    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error saving file: {str(e)}"
        )
    finally:
        file.file.close()
    
    background_tasks.add_task(run_ingestion, target_path=file_path)

    return {
        "message": f"Successfully uploaded {file.filename}. Ingestion started in background.",
        "filename": file.filename,
        "status": "processing"
    }

from app.services.document_service import delete_document_by_name

@router.delete("/{filename}", summary="Delete document from Qdrant and storage")
async def delete_document(filename: str, _: None = Depends(verify_admin)):
    file_path = os.path.join(DATA_DIR, filename)
    if not os.path.exists(file_path):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document '{filename}' not found."
        )

    try:
        result = delete_document_by_name(filename)
        return {
            "message": f"Successfully deleted '{filename}' from vector store and disk.",
            "details": result
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete document: {str(e)}"
        )