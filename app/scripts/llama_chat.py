from llama_index.core import (SimpleDirectoryReader, VectorStoreIndex)
from llama_index.core.schema import NodeWithScore
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.core.response_synthesizers import get_response_synthesizer
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.core.retrievers import QueryFusionRetriever
from llama_index.core import StorageContext
from llama_index.core.postprocessor import LongContextReorder
from llama_index.vector_stores.qdrant import QdrantVectorStore
from app.db.qdrant import client as qdrant_client
from llama_index.postprocessor.jinaai_rerank import JinaRerank

from openai import AsyncOpenAI
import os
from app.core.config import settings
import app.ai.llama
from app.ai.context_selector import select_context

docstore_path = "./storage/docstore/docstore.json"
if not os.path.exists(docstore_path):
    raise FileNotFoundError("Docstore not found. Please run `python scripts/ingest_docs.py` first.")

docstore = SimpleDocumentStore.from_persist_path(docstore_path)

vector_store = QdrantVectorStore(client=qdrant_client, collection_name="fastapi_documents")

storage_context = StorageContext.from_defaults(docstore=docstore, vector_store=vector_store)

index = VectorStoreIndex.from_vector_store(vector_store=vector_store, storage_context=storage_context)

all_nodes = list(docstore.docs.values())
leaf_nodes = [node for node in all_nodes if getattr(node, "child_nodes", None) is None]
for idx, node in enumerate(leaf_nodes):
    node.metadata["node_index"] = idx

bm25_docstore = SimpleDocumentStore()
bm25_docstore.add_documents(leaf_nodes)

bm25_retriever = BM25Retriever.from_defaults(
    docstore=bm25_docstore,
    similarity_top_k=10
)

dense_retriever = index.as_retriever(similarity_top_k=10)

fusion_retriever = QueryFusionRetriever(
    retrievers=[
        dense_retriever,
        bm25_retriever
    ],
    similarity_top_k=10,
    num_queries=1,
    mode="reciprocal_rerank",
    use_async=False
)

#framework postprocessor
reranker = JinaRerank(
    api_key=settings.JINA_API_KEY,
    top_n=5,
    model="jina-reranker-v2-base-multilingual"
)

context_reorder = LongContextReorder()
response_synthesizer = get_response_synthesizer(response_mode="compact")

client = AsyncOpenAI(
    api_key=settings.GROQ_API_KEY,
    base_url="https://api.groq.com/openai/v1"
)

MODEL = "openai/gpt-oss-120b"
conversation = []

async def contextualize_question(question: str) -> str:
    if not conversation:
        return question
    
    history = "\n".join(
        f"{message['role']}: {message['content']}"
        for message in conversation
    )

    prompt = f"""
Given the conversation below, rewrite the user's latest question
into a standalone question that can be understood without the
conversation history.

Do not answer the question.

Conversation:
{history}

Latest question:
{question}

Standalone question:
"""

    reponse = await client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0
    )

    return response.choices[0].message.content.strip()


#Retrieve parent context
async def retrieve_parent_context(query: str) -> list[NodeWithScore]:

    fused_nodes = fusion_retriever.retrieve(query)

    reranked_nodes = reranker.postprocess_nodes(nodes=fused_nodes, query_str=query)

    print("Reranked top hits: ")
    for rank, node_with_score in enumerate(reranked_nodes, start=1):
        print(
            f"{rank}. Score: {node_with_score.score: .4f} | "
            f"File: {node_with_score.node.metadata.get('file_name')}"
        )
    
    parent_nodes = {}

    for node_with_score in reranked_nodes:
        leaf_node =  node_with_score.node
        score = node_with_score.score

        if leaf_node.parent_node:
            parent_id = leaf_node.parent_node.node_id
            parent_doc = docstore.get_document(parent_id)

            parent_nodes[parent_id] = NodeWithScore(
                node = parent_doc,
                score = score
            )

        else: 
            parent_nodes[leaf_node.node_id] = NodeWithScore(
                node = leaf_node,
                score = score
            )
    return list(parent_nodes.values())

    # print("\nFused:")
    # for rank, result in enumerate(fused_results, start=1):
    #     node = result.node
    #     print(
    #         rank,
    #         node.metadata.get("node_index"),
    #         node.metadata.get("file_name"),
    #         result.score
    #     )
    
    # documents = [
    #     leaf_nodes[index].text
    #     for index in fused_indices
    # ]

    # reranked = await rerank(query=query, documents=documents, top_n=10)

    # reranked_documents = [
    #     documents[item["index"]]
    #     for item in reranked["results"]
    # ]

    # relevance_scores = [
    #     item["relevance_score"]
    #     for item in reranked["results"]
    # ]

    # query_embeddings = embed_query(query)
    # doc_embeddings = embed_documents(reranked_documents)

    # selected_indices = mmr(
    #     query_embeddings=query_embeddings,
    #     document_embeddings=doc_embeddings,
    #     relevance_scores=relevance_scores,
    #     top_n=3,
    #     lambda_param=0.7,
    # )

    # parent_nodes = {}

    # for selected_index in selected_indices:
    #     item = reranked["results"][selected_index]
    #     fused_index = fused_indices[item["index"]]
    #     node = leaf_nodes[fused_index]
    #     print(
    #         f"MMR selected: "
    #         f"Reranker score = {item['relevance_score']:.4f} | "
    #         f"Source = {node.metadata.get('file_name')}"
    #     )

    #     if node.parent_node:
    #         parent_id = node.parent_node.node_id
    #         parent = docstore.get_document(parent_id)

    #         parent_nodes[parent_id] = NodeWithScore(
    #             node=parent,
    #             score=item["relevance_score"]
    #         )
    #     else:
    #         parent_nodes[node.node_id] = NodeWithScore(
    #             node=node,
    #             score=item["relevance_score"]
    #         )
    
    # return list(parent_nodes.values())

def generate_answer(question: str, context_nodes: list[NodeWithScore]):
    response = response_synthesizer.synthesize(question, nodes=context_nodes)
    return str(response)

async def run_rag(question: str):
    #rewrite ques using convo history
    standalone_question = await contextualize_question(question)

    #retrieve relevant parent nodes
    context_nodes = await retrieve_parent_context(standalone_question)

    context_nodes = select_context(context_nodes, max_tokens=2000)
    context_nodes = context_reorder.postprocess_nodes(context_nodes)

    # print("\nBefore LongContextReorder:")

    # for rank, node in enumerate(context_nodes, start=1):
    #     print(
    #         rank,
    #         node.score,
    #         node.node.metadata.get("file_name")
    #     )

    # context_nodes = context_reorder.postprocess_nodes(context_nodes)

    # print("\nAfter LongContextReorder:")

    # for rank, node in enumerate(context_nodes, start=1):
    #     print(
    #         rank,
    #         node.score,
    #         node.node.metadata.get("file_name")
    #     )

    #Generate grounded ans
    answer = generate_answer(standalone_question, context_nodes)

    #build context text
    context = "\n\n".join(
        node.node.get_content()
        for node in context_nodes
    )

    sources = []

    for node in context_nodes:
        source = node.node.metadata.get("file_name")

        if source and source not in sources:
            sources.append(source)
    
    return {
        "question": question,
        "standalone_query": standalone_question,
        "context": context,
        "sources": sources,
        "answer": answer
    }

#Chat loop
async def main():
    print("\n====================================")
    print("   FastAPI Documentation Chatbot")
    print("====================================")
    print("Ask questions about the documentation.")
    print("Type 'exit' to quit.\n")

    while True:
        question = input("You: ").strip()

        if question.lower() == "exit":
            break

        if not question:
            continue

        result = await run_rag(question)

        print(f"\nRetrieval query: {result['standalone_query']}")
        print(f"\nAssistant: {result['answer']}\n")

        #Store convo
        conversation.append(
            {
                "role": "user",
                "content": question,
            }
        )

        conversation.append(
            {
                "role": "assistant",
                "content": result["answer"],
            }
        )

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
