from app.db.qdrant import client as qdrant_client, recreate_collection
from llama_index.core.node_parser import HierarchicalNodeParser
import os
import app.ai.llama
from app.observability.langfuse import langfuse
from llama_index.core import SimpleDirectoryReader, StorageContext, VectorStoreIndex
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.vector_stores.qdrant import QdrantVectorStore
from app.core.exceptions import IngestionError

def run_ingestion(data_dir: str = "data"):
    print("--- Starting Persistent Ingestion ---")

    with langfuse.start_as_current_observation(
        name="ingest-pipeline",
        as_type="chain",
        input={
            "data_dir": data_dir
        }
    ) as trace:

        try:
            recreate_collection()


            with langfuse.start_as_current_observation(
                name="parse-and-chunk",
                as_type="chain",
                input={
                    "data_dir": data_dir,
                    "required_exts": [".md"]
                }
            ) as chunk_obs:

                documents = SimpleDirectoryReader(data_dir, recursive=True, required_exts=[".md"]).load_data()
                print(f"Loaded {len(documents)} source documents")

                node_parser = HierarchicalNodeParser.from_defaults(chunk_sizes=[500, 100])
                nodes = node_parser.get_nodes_from_documents(documents)

                leaf_nodes = [node for node in nodes if node.child_nodes is None]

                for index, node in enumerate(leaf_nodes):
                    node.metadata["node_index"] = index
    
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
                    "persist_dir": "./storage/docstore"
                }
            ) as persist_obs:

                docstore = SimpleDocumentStore()
                docstore.add_documents(nodes)

                docstore_dir = "./storage/docstore"
                os.makedirs(docstore_dir, exist_ok=True)
                docstore_path = os.path.join(docstore_dir, "docstore.json")
                docstore.persist(persist_path=docstore_path)
                print(f"Docstore persisted to {docstore_path}")

                persist_obs.update(
                    output={
                        "docstore_path": docstore_path,
                        "nodes_stored": len(nodes)
                    }
                )

            
            with langfuse.start_as_current_observation(
                name="qdrant-index",
                as_type="chain",
                input={
                    "collection_name": "fastapi_documents",
                    "leaf_nodes": len(leaf_nodes)
                }
            ) as index_obs:

                vector_store = QdrantVectorStore(client=qdrant_client, collection_name="fastapi_documents")

                storage_context = StorageContext.from_defaults(docstore=docstore, vector_store=vector_store)

                VectorStoreIndex(
                    leaf_nodes,
                    storage_context=storage_context,
                )
                print("--- Ingestion Complete: Leaf Nodes Embedded and Stored in Qdrant ---")    

                index_obs.update(
                    output={
                        "indexed_leaf_nodes": len(leaf_nodes),
                        "collection": "fastapi_documents"
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