import os
from qdrant_client.http import models
from llama_index.core.storage.docstore import SimpleDocumentStore
from app.db.qdrant import client as qdrant_client

COLLECTION_NAME = "fastapi_documents"
DOCSTORE_PATH = "./storage/docstore/docstore.json"
DATA_DIR = "./data"

def delete_document_by_name(filename: str) -> dict:
    # 1. Delete vectors from Qdrant using payload filter
    qdrant_client.delete(
        collection_name=COLLECTION_NAME,
        points_selector=models.FilterSelector(
            filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="file_name",
                        match=models.MatchValue(value=filename)
                    )
                ]
            )
        )
    )

    # 2. Delete nodes from LlamaIndex Docstore
    deleted_nodes_count = 0
    if os.path.exists(DOCSTORE_PATH):
        docstore = SimpleDocumentStore.from_persist_path(DOCSTORE_PATH)
        nodes_to_remove = [
            node_id for node_id, node in docstore.docs.items()
            if node.metadata.get("file_name") == filename
        ]
        
        for node_id in nodes_to_remove:
            docstore.delete_document(node_id, raise_error=False)
            deleted_nodes_count += 1
            
        docstore.persist(DOCSTORE_PATH)

    # 3. Remove file from disk
    file_path = os.path.join(DATA_DIR, filename)
    file_deleted = False
    if os.path.exists(file_path):
        os.remove(file_path)
        file_deleted = True

    return {
        "filename": filename,
        "file_deleted": file_deleted,
        "nodes_removed": deleted_nodes_count
    }