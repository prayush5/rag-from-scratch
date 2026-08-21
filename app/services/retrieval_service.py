import os
import asyncio
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.core.schema import NodeWithScore
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.core.retrievers import QueryFusionRetriever
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.postprocessor.jinaai_rerank import JinaRerank

from app.core.config import settings
from app.db.qdrant import client as qdrant_client
from app.db.qdrant import async_client
from app.core.exceptions import RetrievalError
from app.observability.langfuse import langfuse

class RetrievalService:
    def __init__(self):
        docstore_path = "./storage/docstore/docstore.json"
        if not os.path.exists(docstore_path):
            raise FileNotFoundError("Docstore not found. Please run `python scripts/ingest_docs.py` first.")

        self.docstore = SimpleDocumentStore.from_persist_path(docstore_path)
        self.vector_store = QdrantVectorStore(client=qdrant_client, aclient=async_client, collection_name="fastapi_documents")
        self.storage_context = StorageContext.from_defaults(docstore=self.docstore, vector_store=self.vector_store)
        self.index = VectorStoreIndex.from_vector_store(vector_store=self.vector_store, storage_context=self.storage_context)

        all_nodes = list(self.docstore.docs.values())
        leaf_nodes = [node for node in all_nodes if getattr(node, "child_nodes", None) is None]
        for idx, node in enumerate(leaf_nodes):
            node.metadata["node_index"] = idx
        
        bm25_docstore = SimpleDocumentStore()
        bm25_docstore.add_documents(leaf_nodes)

        self.bm25_retriever = BM25Retriever.from_defaults(docstore=bm25_docstore, similarity_top_k=10)
        self.dense_retriever = self.index.as_retriever(similarity_top_k=10)

        self.fusion_retriever = QueryFusionRetriever(
            retrievers=[self.dense_retriever, self.bm25_retriever],
            similarity_top_k=10,
            num_queries=1,
            mode="reciprocal_rerank",
            use_async=True,
        )

        self.reranker = JinaRerank(
            api_key=settings.JINA_API_KEY,
            top_n=5,
            model="jina-reranker-v2-base-multilingual",
        )

    async def retrieve_parent_context(self, query: str) -> list[NodeWithScore]:
        with langfuse.start_as_current_observation(
            name="retrieval",
            as_type="chain",
            input={"query": query}
        ) as observation:
            try:
                fused_nodes = await self.fusion_retriever.aretrieve(query)
                reranked_nodes = await self.reranker.apostprocess_nodes(
                    nodes=fused_nodes,
                    query_str=query,
                )

                parent_nodes: dict[str, NodeWithScore] = {}

                for node_with_score in reranked_nodes:
                    leaf_node = node_with_score.node
                    score = node_with_score.score

                    if leaf_node.parent_node:
                        parent_id = leaf_node.parent_node.node_id
                        
                        # Preserve the score of the highest-ranked child for each parent
                        if parent_id not in parent_nodes:
                            parent_doc = self.docstore.docs.get(parent_id)
                            if parent_doc:
                                parent_nodes[parent_id] = NodeWithScore(node=parent_doc, score=score)
                            else:
                                parent_nodes[leaf_node.node_id] = NodeWithScore(node=leaf_node, score=score)
                    elif leaf_node.node_id not in parent_nodes:
                        parent_nodes[leaf_node.node_id] = NodeWithScore(node=leaf_node, score=score)

                result = list(parent_nodes.values())
                
                observation.update(
                    output={
                        "fused_nodes": len(fused_nodes),
                        "reranked_nodes": len(reranked_nodes),
                        "parent_nodes": len(result)
                    }
                )
            
                return result
        
            except Exception as ex:
                raise RetrievalError("Failed to retrieve parent context") from ex