from typing import Annotated, TypedDict, Optional
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_groq import ChatGroq

from app.core.config import settings
from app.ai.agent_tools import search_documentation

agent_llm = ChatGroq(
    model=settings.LLM_MODEL,
    api_key=settings.GROQ_API_KEY,
    temperature=0,
)

ROUTER_SYSTEM_PROMPT = """You have access to a documentation search tool. Call it only
if the question is about the indexed technical documentation (FastAPI, Python,
SQLAlchemy, Spring Boot, etc.). For general knowledge questions (math, facts,
capitals, etc.), answer directly without calling the tool.

When calling the search tool, phrase the query as a standalone question that
doesn't depend on earlier turns — resolve pronouns like "it" or "that" into
the actual subject based on the conversation so far."""

STRICT_CONTEXT_PROMPT = """Answer the question using ONLY the documentation excerpt below.
If the excerpt does not fully answer the question, say exactly what it does cover and
explicitly state what's missing. Do not add anything not present in the excerpt.

Documentation excerpt:
{context}

Question: {question}
"""

GENERAL_KNOWLEDGE_PROMPT = """Answer this question using your own general knowledge,
concisely, in a few sentences.

Question: {question}
"""

class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    question: str
    standalone_question: Optional[str]
    tool_output: Optional[str]
    doc_answer: Optional[str]
    needs_supplement: bool
    general_answer: Optional[str]
    sources: list[str]

#decides whether the question needs documentation search
router_llm = agent_llm.bind_tools([search_documentation])

async def router_node(state: AgentState):
    response = await router_llm.ainvoke(
        [SystemMessage(content=ROUTER_SYSTEM_PROMPT)] + state["messages"]
    )
    return {"messages":[response]}

def router_after_router(state: AgentState):
    last = state["messages"][-1]
    if getattr(last, "tool_calls", None):
        return "tools"
    return END

#actually runs search_documentation if the router called it
tool_node = ToolNode([search_documentation])

#pulls raw tool text + sources into state
async def extract_tool_output(state: AgentState):
    tool_message = state["messages"][-1]
    ai_with_call = state["messages"][-2] if len(state["messages"]) >= 2 else None

    standalone_question = state["question"]
    if ai_with_call is not None and getattr(ai_with_call, "tool_calls", None):
        standalone_question = ai_with_call.tool_calls[0]["args"].get("query", state["question"])

    text = tool_message.content if isinstance(tool_message.content, str) else str(tool_message.content)
    sources = set()
    for line in text.split("---"):
        line = line.strip()
        if line.startswith("[Source:"):
            sources.add(line.split("[Source:")[1].split("]")[0].strip())

    return {"tool_output": text, "sources": list(sources), "standalone_question": standalone_question}

def _extract_text(content) -> str:
    if isinstance(content, list):
        return "".join(
            block.get("text", "") if isinstance(block, dict) else str(block)
            for block in content
        )
    return content

#answers ONLY from the retrieved documentation
async def strict_answer_node(state: AgentState):
    question = state.get("standalone_question") or state["question"]
    prompt = STRICT_CONTEXT_PROMPT.format(context=state["tool_output"], question=question)
    response = await agent_llm.ainvoke([HumanMessage(content=prompt)])
    return {"doc_answer": _extract_text(response.content)}

#one-word verdict: is the doc answer complete?
async def decide_supplement_node(state: AgentState):
    question = state.get("standalone_question") or state["question"]
    check_prompt = (
        f"Question: {question}\n\n"
        f"Documentation-based answer: {state['doc_answer']}\n\n"
        "Does this answer fully address the question, or is information missing "
        "that the documentation simply didn't cover? Reply with exactly one word: "
        "COMPLETE or INCOMPLETE."
    )
    response = await agent_llm.ainvoke([HumanMessage(content=check_prompt)])
    needs = "INCOMPLETE" in _extract_text(response.content).upper()
    return {"needs_supplement": needs}

def router_after_supplement(state: AgentState):
    return "general_answer" if state["needs_supplement"] else "finalize"

#only runs if the doc answer was incomplete
async def general_answer_node(state: AgentState):
    question = state.get("standalone_question") or state["question"]
    prompt = GENERAL_KNOWLEDGE_PROMPT.format(question=question)
    response = await agent_llm.ainvoke([HumanMessage(content=prompt)])
    return {"general_answer": _extract_text(response.content)}

#glues doc_answer + general_answer with a FIXED template
# (this is the actual enforcement point — no LLM involved in the labeling)
async def finalize_node(state: AgentState):
    if state.get("doc_answer"):
        final = f"** According to the documentation:**\n{state['doc_answer']}"
        if state.get("general_answer"):
            final += f"\n\n**More generally (not from documents):**\n{state['general_answer']}"
    else:
        final = _extract_text(state["messages"][-1].content)

    return {"messages": [AIMessage(content=final)]}    
    

graph_builder = StateGraph(AgentState)
graph_builder.add_node("router", router_node)
graph_builder.add_node("tools", tool_node)
graph_builder.add_node("extract_tool_output", extract_tool_output)
graph_builder.add_node("strict_answer", strict_answer_node)
graph_builder.add_node("decide_supplement", decide_supplement_node)
graph_builder.add_node("general_answer", general_answer_node)
graph_builder.add_node("finalize", finalize_node)

graph_builder.add_edge(START, "router")
graph_builder.add_conditional_edges("router", router_after_router, {"tools": "tools", END: END})
graph_builder.add_edge("tools", "extract_tool_output")
graph_builder.add_edge("extract_tool_output", "strict_answer")
graph_builder.add_edge("strict_answer", "decide_supplement")
graph_builder.add_conditional_edges("decide_supplement", router_after_supplement, {"general_answer": "general_answer", "finalize": "finalize"})
graph_builder.add_edge("general_answer", "finalize")
graph_builder.add_edge("finalize", END)

def compile_agent(checkpointer):
    return graph_builder.compile(checkpointer=checkpointer)
