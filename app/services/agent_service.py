from typing import AsyncGenerator, Any
from langchain_core.messages import HumanMessage

STATUS_MESSAGES = {
    "router": "Thinking about how to answer...",
    "tools": "Accessing documentation...",
    "extract_tool_output": "Reviewing search results...",
    "strict_answer": "Reading through documentation...",
    "decide_supplement": "Checking if anything is missing...",
    "general_answer": "Filling in from general knowledge...",
    "finalize": "Putting together the answer..."
}

async def stream_agent_answer(agent, question: str, thread_id: str) -> AsyncGenerator[tuple[str, Any], None]:
    config = {
        "configurable": {"thread_id": thread_id},
        "recursion_limit": 10,
    }

    seen_statuses = set()

    try:

        async for event in agent.astream_events(
            {
                "messages": [HumanMessage(content=question)],
                "question": question,
                "standalone_question": None,
                "tool_output": None,
                "doc_answer": None,
                "needs_supplement": False,
                "general_answer": None,
                "sources": []
            },
            config=config,
            version="v2"
        ):
            kind = event["event"]
            name = event.get("name")

            if kind == "on_chain_start" and name in STATUS_MESSAGES and name not in seen_statuses:
                seen_statuses.add(name)
                yield("status", STATUS_MESSAGES[name])
            
            if kind == "on_chain_end" and name == "LangGraph":
                output = event["data"]["output"]
                answer = output["messages"][-1].content
                sources = output.get("sources", [])
                yield("token", answer)
                yield("metadata", {"sources": sources})

    except Exception as ex:
        yield ("token", f"\n\n⚠ Something went wrong while generating a response: {ex}")
        yield ("metadata", {"sources": []})