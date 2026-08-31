from app.db.qdrant import client as qdrant_client, init_collection
from llama_index.core.node_parser import HierarchicalNodeParser
import os, hashlib, uuid
import app.ai.llama
from app.observability.langfuse import langfuse
from llama_index.core import SimpleDirectoryReader, StorageContext, VectorStoreIndex
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.vector_stores.qdrant import QdrantVectorStore
from app.core.exceptions import IngestionError
from llama_index.readers.file import PDFReader
from qdrant_client.http import models

COLLECTION_NAME = "fastapi_documents"
DOCSTORE_DIR = "./storage/docstore"
DOCSTORE_PATH = os.path.join(DOCSTORE_DIR, "docstore.json")

def purge_file_records(filename: str, docstore_path: str):
    try:
        qdrant_client.delete(
            collection_name=COLLECTION_NAME,
            points_selector=models.FilterSelector(
                filter=models.Filter(
                    must=[models.FieldCondition(key="file_name", match=models.MatchValue(value=filename))]
                )
            )
        )
    except Exception as e:
        print(f"Qdrant purge warning for {filename}: {str(e)}")

    if os.path.exists(docstore_path):
        docstore = SimpleDocumentStore.from_persist_path(docstore_path)
        matching_ids = [
            node_id for node_id, node in docstore.docs.items()
            if node.metadata.get("file_name") == filename
        ]

        if matching_ids:
            for node_id in matching_ids:
                docstore.delete_document(node_id, raise_error=False)
            docstore.persist(persist_path=docstore_path)
            print(f"Purged {len(matching_ids)} docstore node(s) for '{filename}'")
        else:
            print(f"No matching docstore nodes found to purge for '{filename}'")


def run_ingestion(data_dir: str = "data", target_path: str = None):
    print("--- Starting Persistent Ingestion ---")

    with langfuse.start_as_current_observation(
        name="ingest-pipeline",
        as_type="chain",
        input={
            "data_dir": data_dir,
            "target_path": target_path
        }
    ) as trace:

        try:
            init_collection()
            file_extractor = {".pdf": PDFReader()}

            with langfuse.start_as_current_observation(
                name="parse-and-chunk",
                as_type="chain",
                input={
                    "target_path": target_path or data_dir,
                    "required_exts": [".md", ".pdf"]
                }
            ) as chunk_obs:

                if target_path and os.path.isfile(target_path):
                    print(f"Ingesting single target file: {target_path}")
                    reader = SimpleDirectoryReader(input_files=[target_path], file_extractor=file_extractor)
                else:
                    print(f"Scanning directory: {data_dir}")
                    reader = SimpleDirectoryReader(data_dir, recursive=True, required_exts=[".md", ".pdf"], file_extractor=file_extractor)
                    
                documents = reader.load_data()
                if not documents:
                    print("No documents found to process")
                    return {"status": "no_documents"}

                print(f"Loaded {len(documents)} source documents")

                unique_filenames = set()
                for doc in documents:
                    fname = doc.metadata.get("file_name")
                    if not fname and doc.metadata.get("file_path"):
                        fname = os.path.basename(doc.metadata["file_path"])
                    elif not fname and target_path:
                        fname = os.path.basename(target_path)
                    
                    if fname:
                        doc.metadata["file_name"] = fname
                        unique_filenames.add(fname)

                for fname in unique_filenames:
                    purge_file_records(fname, DOCSTORE_PATH)

                node_parser = HierarchicalNodeParser.from_defaults(chunk_sizes=[500, 100])
                nodes = node_parser.get_nodes_from_documents(documents)

                leaf_nodes = [node for node in nodes if node.child_nodes is None]
    
                print(f"Total nodes: {len(nodes)} | Leaf nodes: {len(leaf_nodes)}")

                chunk_obs.update(
                    output={
                        "raw_documents": len(documents),
                        "total_nodes": len(nodes),
                        "leaf_nodes": len(leaf_nodes),
                    }
                )

            
            with langfuse.start_as_current_observation(
                name="docstore-persist",
                as_type="chain",
                input={
                    "persist_dir": DOCSTORE_DIR
                }
            ) as persist_obs:

                os.makedirs(DOCSTORE_DIR, exist_ok=True)

                if os.path.exists(DOCSTORE_PATH):
                    docstore = SimpleDocumentStore.from_persist_path(DOCSTORE_PATH)
                else:
                    docstore = SimpleDocumentStore()
                
                docstore.add_documents(nodes)
                docstore.persist(persist_path=DOCSTORE_PATH)
                print(f"Docstore persisted to {DOCSTORE_PATH}")

                persist_obs.update(
                    output={
                        "docstore_path": DOCSTORE_PATH,
                        "nodes_stored": len(nodes)
                    }
                )

            
            with langfuse.start_as_current_observation(
                name="qdrant-index",
                as_type="chain",
                input={
                    "collection_name": COLLECTION_NAME,
                    "leaf_nodes": len(leaf_nodes)
                }
            ) as index_obs:

                vector_store = QdrantVectorStore(client=qdrant_client, collection_name=COLLECTION_NAME)

                storage_context = StorageContext.from_defaults(docstore=docstore, vector_store=vector_store)

                VectorStoreIndex(
                    leaf_nodes,
                    storage_context=storage_context,
                )
                print("--- Ingestion Complete: Leaf Nodes Embedded and Stored in Qdrant ---")    

                index_obs.update(
                    output={
                        "indexed_leaf_nodes": len(leaf_nodes),
                        "collection": COLLECTION_NAME
                    }
                )

            res = {
                    "documents_processed": len(documents),
                    "nodes_created": len(nodes),
                    "leaf_nodes_created": len(leaf_nodes)
            }

            trace.update(output=res)
            return res
            
        except Exception as ex:
            raise IngestionError(f"Ingestion failed: {str(ex)}") from ex
        finally:
            langfuse.flush()


if __name__ == "__main__":
    run_ingestion()