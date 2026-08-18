from llama_index.core.schema import NodeWithScore

def estimate_tokens(text: str) -> int:
    return len(text)//4

def select_context(nodes: list[NodeWithScore], max_tokens: int = 2000) -> list[NodeWithScore]:
    selected = []
    total_tokens = 0

    for node in nodes:
        text = node.node.get_content()
        tokens = estimate_tokens(text)

        if total_tokens + tokens > max_tokens:
            continue

        selected.append(node)
        total_tokens += tokens

    print(
        f"Context selection: "
        f"{len(nodes)} → {len(selected)} nodes "
        f"(~{total_tokens} tokens)"
    )

    return selected
