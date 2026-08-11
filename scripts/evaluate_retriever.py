import json

from llama_index.core import (SimpleDirectoryReader, VectorStoreIndex)
from llama_index.core.node_parser import HierarchicalNodeParser

import app.ai.llama

with open("data/evaluation/rag_eval.json", "r") as f:
    eval_cases = json.load(f)

documents = SimpleDirectoryReader("data", recursive=True, required_exts=[".md"]).load_data()

print(f"Documents: {len(documents)}")

node_parser = HierarchicalNodeParser.from_defaults(chunk_sizes=[500, 100])

nodes = node_parser.get_nodes_from_documents(documents)

leaf_nodes = [
    node
    for node in nodes
    if node.child_nodes is None
]

print(f"All nodes: {len(nodes)}")
print(f"Leaf nodes: {len(leaf_nodes)}")

index = VectorStoreIndex(leaf_nodes)

retriever = index.as_retriever(similarity_top_k=3)

#evaluate retrieval
for case in eval_cases:
    question = case["question"]
    expected_sources = set(case["expected_sources"])

    results = retriever.retrieve(question)

    retrieved_sources = [
        result.node.metadata.get("file_name")
        for result in results
    ]

    print("\n" + "=" * 60)
    print(f"Question: {question}")
    print(f"Expected: {expected_sources}")

    print("\nRetrieved:")

    for rank, source in enumerate(retrieved_sources, start=1):
        score = results[rank - 1].score
        print(
            f"{rank}. {source} "
            f"(score={score:.4f})"
        )
