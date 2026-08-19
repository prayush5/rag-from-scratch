import os
import app.ai.llama
import asyncio

from openai import AsyncOpenAI
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.core.schema import NodeWithScore
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.core.response_synthesizers import get_response_synthesizer
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.core.retrievers import QueryFusionRetriever
from llama_index.core.postprocessor import LongContextReorder
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.postprocessor.jinaai_rerank import JinaRerank

from app.core.config import settings
from app.db.qdrant import client as qdrant_client
from app.ai.context_selector import select_context
from app.ai.query_rewriter import QueryRewriter
from app.schemas.chat import Message, ChatResponse
from app.core.exceptions import RetrievalError, GenerationError
from app.observability.langfuse import langfuse

class RAGService:
    def __init__(self):
        docstore_path = "./storage/docstore/docstore.json"
        if not os.path.exists(docstore_path):
            raise FileNotFoundError("Docstore not found. Please run `python scripts/ingest_docs.py` first.")

        self.docstore = SimpleDocumentStore.from_persist_path(docstore_path)
        self.vector_store = QdrantVectorStore(client=qdrant_client, collection_name="fastapi_documents")
        self.storage_context = StorageContext.from_defaults(docstore=self.docstore, vector_store=self.vector_store)
        self.index = VectorStoreIndex.from_vector_store(vector_store=self.vector_store, storage_context=self.storage_context)

        # Re-index leaf nodes for BM25
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
            use_async=False,
        )

        self.reranker = JinaRerank(
            api_key=settings.JINA_API_KEY,
            top_n=5,
            model="jina-reranker-v2-base-multilingual",
        )

        self.context_reorder = LongContextReorder()
        self.response_synthesizer = get_response_synthesizer(response_mode="compact")
        self.llm_client = AsyncOpenAI(
            api_key=settings.GROQ_API_KEY,
            base_url="https://api.groq.com/openai/v1",
        )
        self.model = "openai/gpt-oss-120b"

        self.query_rewriter = QueryRewriter(
            llm_client=self.llm_client,
            model=self.model
        )

    async def retrieve_parent_context(self, query: str) -> list[NodeWithScore]:
        with langfuse.start_as_current_observation(
            name="retrieval",
            as_type="chain",
            input={"query": query}
        ) as observation:
            try:
                fused_nodes = await asyncio.to_thread(self.fusion_retriever.retrieve, query)
                reranked_nodes = await self.reranker.apostprocess_nodes(
                    nodes=fused_nodes,
                    query_str=query,
                )

                parent_nodes = {}

                for node_with_score in reranked_nodes:
                    leaf_node = node_with_score.node
                    score = node_with_score.score

                    if leaf_node.parent_node:
                        parent_id = leaf_node.parent_node.node_id
                        parent_doc = self.docstore.get_document(parent_id)
                        parent_nodes[parent_id] = NodeWithScore(node=parent_doc, score=score)
                    else:
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
                raise RetrievalError(
                    "Failed to retrieve parent context"
                ) from ex

    async def answer_question(self, question: str, history: list[Message]) -> ChatResponse:
        with langfuse.start_as_current_observation(
            name="rag-chat",
            as_type="chain",
            input={
                "question": question,
                "history": [msg.model_dump() for msg in history]
            }
        ) as trace:

            standalone_question = await self.query_rewriter.rewrite(question, history)
            retrieved_parent_nodes = await self.retrieve_parent_context(standalone_question)

            # Instrument context selection window
            with langfuse.start_as_current_observation(
                name="context-selection",
                as_type="chain",
                input={
                    "input_nodes": len(retrieved_parent_nodes),
                    "max_tokens": 2000
                }
            ) as context_obs:
                selected_nodes = select_context(retrieved_parent_nodes, max_tokens=2000)
                estimated_tokens = sum(len(node.node.get_content()) // 4 for node in selected_nodes)

                context_obs.update(
                    output={
                        "selected_nodes": len(selected_nodes),
                        "estimated_tokens": estimated_tokens
                    }
                )

            context_nodes = await self.context_reorder.apostprocess_nodes(selected_nodes)

            # --- Response Generation Observation ---
            with langfuse.start_as_current_observation(
                name="response-generation",
                as_type="generation",
                input={
                    "prompt_query": standalone_question,
                    "context_snippets": [node.node.get_content() for node in context_nodes],
                },
                model=self.model,
            ) as gen_obs:
                try:
                    raw_response = await self.response_synthesizer.asynthesize(
                        standalone_question, 
                        nodes=context_nodes
                    )
                except Exception as ex:
                    raise GenerationError("Failed to generate response") from ex

                answer = str(raw_response)
                gen_obs.update(output=answer)

            context = "\n\n".join(node.node.get_content() for node in context_nodes)
            sources = []
            for node in context_nodes:
                source = node.node.metadata.get("file_name")
                if source and source not in sources:
                    sources.append(source)
        
            trace.update(
                output={
                    "standalone_query": standalone_question,
                    "answer": answer,
                    "sources": sources
                }
            )
        
            return ChatResponse(
                question=question,
                standalone_query=standalone_question,
                answer=answer,
                sources=sources,
                context=context,
            )