from langchain_core.tools import tool
from app.services.retrieval_service import RetrievalService
from app.ai.context_selector import select_context
from langchain_core.messages import HumanMessage
from langchain_groq import ChatGroq
from app.core.config import settings

_retrieval_service = RetrievalService()

_tool_llm = ChatGroq(
    model=settings.LLM_MODEL,
    api_key=settings.GROQ_API_KEY,
    temperature=0,
)

QUERY_CORRECTION_PROMPT = """Fix any obvious typos or spelling mistakes in this technical query. 
Return ONLY the corrected query without extra commentary.

Query: {query}"""

@tool
async def search_documentation(query: str):
    """Search the indexed technical documentation (FastAPI, Python, Spring Boot, and other
    ingested docs) for information relevant to the query. Use this whenever the user asks
    a question that could plausibly be answered by internal documentation, rather than
    answering from general knowledge."""

    nodes = await _retrieval_service.retrieve_parent_context(query)

    if not nodes or nodes[0].score < 0.15:
        clean_res = await _tool_llm.ainvoke(
            [HumanMessage(content=QUERY_CORRECTION_PROMPT.format(query=query))]
        )
        cleaned_query = clean_res.content.strip()
        if cleaned_query.lower() != query.lower():
            corrected_nodes = await _retrieval_service.retrieve_parent_context(cleaned_query)
            if corrected_nodes and corrected_nodes[0].score > (nodes[0].score if nodes else 0):
                nodes = corrected_nodes
    
    if not nodes:
        return "No relevant documentation found for the query."

    RELEVANCE_THRESHOLD = 0.25
    relevant_nodes = [n for n in nodes if n.score is not None and n.score >= RELEVANCE_THRESHOLD]

    if not relevant_nodes:
        return "No relevant documents found for this query."
    
    selected_nodes = select_context(relevant_nodes, max_tokens=1200)

    return "\n\n---\n\n".join(
        f"[Source: {n.node.metadata.get('file_name', 'unknown')}]\n{n.node.get_content()}"
        for n in selected_nodes
    )