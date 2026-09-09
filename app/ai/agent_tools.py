from langchain_core.tools import tool
from app.services.retrieval_service import RetrievalService
from app.ai.context_selector import select_context

_retrieval_service = RetrievalService()

@tool
async def search_documentation(query: str):
    """Search the indexed technical documentation (FastAPI, Python, Spring Boot, and other
    ingested docs) for information relevant to the query. Use this whenever the user asks
    a question that could plausibly be answered by internal documentation, rather than
    answering from general knowledge."""

    nodes = await _retrieval_service.retrieve_parent_context(query)
    
    if not nodes:
        return "No relevant documentation found for the query."
    
    selected_nodes = select_context(nodes, max_tokens=1200)

    return "\n\n---\n\n".join(
        f"[Source: {n.node.metadata.get('file_name', 'unknown')}]\n{n.node.get_content()}"
        for n in selected_nodes
    )