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

ROUTER_SYSTEM_PROMPT = """You have access to a documentation search tool that searches
an internal knowledge base covering a variety of topics — technical documentation,
security concepts, and other reference material.

Call search_documentation whenever the question could plausibly be answered by
something in that knowledge base — this includes factual, biographical, historical,
or "trivia-shaped" questions, not just software/technical ones. Err on the side of
searching when in doubt, since the tool will simply report back if nothing relevant
is found.

Only skip the tool for things that clearly cannot be document lookups: basic
arithmetic, casual conversation, or simple greetings.

When calling the search tool, phrase the query as a standalone question that doesn't
depend on earlier turns — resolve pronouns like "it" or "that" into the actual
subject based on the conversation so far."""

STRICT_CONTEXT_PROMPT = """Answer the question using ONLY the documentation excerpt below.
If the excerpt does not fully answer the question, say exactly what it does cover and
explicitly state what's missing. Do not add anything not present in the excerpt.

Documentation excerpt:
{context}

Question: {question}"""

GENERAL_KNOWLEDGE_PROMPT = """Answer this question using your own general knowledge concisely.

Question: {question}"""

class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    question: str
    standalone_question: Optional[str]
    tool_output: Optional[str]
    doc_answer: Optional[str]
    needs_supplement: bool
    general_answer: Optional[str]
    sources: list[str]

router_llm = agent_llm.bind_tools([search_documentation])

async def router_node(state: AgentState):
    response = await router_llm.ainvoke(
        [SystemMessage(content=ROUTER_SYSTEM_PROMPT)] + state["messages"]
    )
    return {"messages": [response]}

def router_after_router(state: AgentState):
    last = state["messages"][-1]
    if getattr(last, "tool_calls", None) and len(last.tool_calls) > 0:
        return "tools"
    return "finalize"

tool_node = ToolNode([search_documentation])

async def extract_tool_output(state: AgentState):
    tool_message = state["messages"][-1]
    ai_with_call = state["messages"][-2] if len(state["messages"]) >= 2 else None

    standalone_question = state.get("question", "")
    if ai_with_call and getattr(ai_with_call, "tool_calls", None):
        standalone_question = ai_with_call.tool_calls[0]["args"].get("query", standalone_question)

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
    return str(content)

async def strict_answer_node(state: AgentState):
    question = state.get("standalone_question") or state["question"]
    prompt = STRICT_CONTEXT_PROMPT.format(context=state["tool_output"], question=question)
    response = await agent_llm.ainvoke([HumanMessage(content=prompt)])
    return {"doc_answer": _extract_text(response.content)}

async def decide_supplement_node(state: AgentState):
    question = state.get("standalone_question") or state["question"]
    check_prompt = (
        f"Question: {question}\n\n"
        f"Documentation answer: {state['doc_answer']}\n\n"
        "Does this fully answer the question? Reply with exactly one word: COMPLETE or INCOMPLETE."
    )
    response = await agent_llm.ainvoke([HumanMessage(content=check_prompt)])
    needs = "INCOMPLETE" in _extract_text(response.content).upper()
    return {"needs_supplement": needs}

def router_after_supplement(state: AgentState):
    return "general_answer" if state["needs_supplement"] else "finalize"

async def general_answer_node(state: AgentState):
    question = state.get("standalone_question") or state["question"]
    prompt = GENERAL_KNOWLEDGE_PROMPT.format(question=question)
    response = await agent_llm.ainvoke([HumanMessage(content=prompt)])
    return {"general_answer": _extract_text(response.content)}

async def finalize_node(state: AgentState):
    if state.get("doc_answer"):
        final = f"**According to the documentation:**\n{state['doc_answer']}"
        if state.get("general_answer"):
            final += f"\n\n**More generally (not from documents):**\n{state['general_answer']}"
    else:
        raw_content = _extract_text(state["messages"][-1].content)
        final = f"**General Knowledge Answer:**\n{raw_content}"

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
graph_builder.add_conditional_edges("router", router_after_router, {"tools": "tools", "finalize": "finalize"})
graph_builder.add_edge("tools", "extract_tool_output")
graph_builder.add_edge("extract_tool_output", "strict_answer")
graph_builder.add_edge("strict_answer", "decide_supplement")
graph_builder.add_conditional_edges("decide_supplement", router_after_supplement, {"general_answer": "general_answer", "finalize": "finalize"})
graph_builder.add_edge("general_answer", "finalize")
graph_builder.add_edge("finalize", END)

def compile_agent(checkpointer):
    return graph_builder.compile(checkpointer=checkpointer)