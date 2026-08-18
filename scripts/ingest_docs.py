from app.db.qdrant import client as qdrant_client, recreate_collection
from llama_index.core.node_parser import HierarchicalNodeParser
import os
import app.ai.llama
from llama_index.core import SimpleDirectoryReader, StorageContext, VectorStoreIndex
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.vector_stores.qdrant import QdrantVectorStore

def run_ingestion():
    print("--- Starting Persistent Ingestion ---")

    recreate_collection()

    documents = SimpleDirectoryReader("data",recursive=True,required_exts=[".md"]).load_data()
    print(f"Loaded {len(documents)} source documents")

    node_parser = HierarchicalNodeParser.from_defaults(chunk_sizes=[500, 100])
    nodes = node_parser.get_nodes_from_documents(documents)

    leaf_nodes = [node for node in nodes if node.child_nodes is None]

    for index, node in enumerate(leaf_nodes):
        node.metadata["node_index"] = index
    
    print(f"Total nodes: {len(nodes)} | Leaf nodes: {len(leaf_nodes)}")

    docstore = SimpleDocumentStore()
    docstore.add_documents(nodes)

    docstore_dir = "./storage/docstore"
    os.makedirs(docstore_dir, exist_ok=True)
    docstore_path = os.path.join(docstore_dir, "docstore.json")
    docstore.persist(persist_path=docstore_path)
    print(f"Docstore persisted to {docstore_path}")

    vector_store = QdrantVectorStore(client=qdrant_client, collection_name="fastapi_documents")

    storage_context = StorageContext.from_defaults(docstore=docstore, vector_store=vector_store)

    VectorStoreIndex(
        leaf_nodes,
        storage_context=storage_context,
    )
    print("--- Ingestion Complete: Leaf Nodes Embedded and Stored in Qdrant ---")    

    return {
        "documents_processed": len(documents),
        "nodes_created": len(nodes),
        "leaf_nodes_created": len(leaf_nodes)
    }


if __name__ == "__main__":
    run_ingestion()