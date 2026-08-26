import json

from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from llama_index.core.node_parser import HierarchicalNodeParser

import app.ai.llama


# --------------------------------------------------
# Load evaluation dataset
# --------------------------------------------------

with open(
    "data/evaluation/rag_eval.json",
    "r",
    encoding="utf-8"
) as f:
    evaluation_data = json.load(f)


# --------------------------------------------------
# Load documents
# --------------------------------------------------

documents = SimpleDirectoryReader(
    "data",
    recursive=True,
    required_exts=[".md"]
).load_data()

print(f"Documents: {len(documents)}")


# --------------------------------------------------
# Hierarchical chunking
# --------------------------------------------------

node_parser = HierarchicalNodeParser.from_defaults(
    chunk_sizes=[500, 100]
)

nodes = node_parser.get_nodes_from_documents(documents)

leaf_nodes = [
    node
    for node in nodes
    if node.child_nodes is None
]

print(f"All nodes: {len(nodes)}")
print(f"Leaf nodes: {len(leaf_nodes)}")


# --------------------------------------------------
# Vector index + retriever
# --------------------------------------------------

index = VectorStoreIndex(leaf_nodes)

retriever = index.as_retriever(
    similarity_top_k=10
)


# --------------------------------------------------
# Helper: remove duplicate sources
# --------------------------------------------------

def unique_sources(results) -> list[str]:
    sources = []
    seen = set()

    for result in results:
        source = result.node.metadata.get("file_name")

        if source not in seen:
            sources.append(source)
            seen.add(source)

    return sources


# --------------------------------------------------
# Evaluation metrics
# --------------------------------------------------

def recall_at_k(
    retrieved_sources: list[str],
    expected_sources: set[str],
    k: int
) -> float:

    retrieved = set(retrieved_sources[:k])

    return float(bool(retrieved & expected_sources))


def reciprocal_rank(
    retrieved_sources: list[str],
    expected_sources: set[str]
) -> float:

    for rank, source in enumerate(
        retrieved_sources,
        start=1
    ):
        if source in expected_sources:
            return 1 / rank

    return 0.0


# --------------------------------------------------
# Run evaluation
# --------------------------------------------------

recall_at_1_scores = []
recall_at_3_scores = []
mrr_scores = []


for item in evaluation_data:

    question = item["question"]
    expected_sources = set(
        item["expected_sources"]
    )

    results = retriever.retrieve(question)

    retrieved_sources = unique_sources(results)

    print("\n" + "=" * 60)
    print(f"Question: {question}")
    print(f"Expected: {expected_sources}")

    print("\nRaw retrieved chunks:")

    for rank, result in enumerate(
        results,
        start=1
    ):
        source = result.node.metadata.get(
            "file_name"
        )

        print(
            f"{rank}. {source} "
            f"(score={result.score:.4f})"
        )

    print("\nRetrieved unique sources:")

    for rank, source in enumerate(
        retrieved_sources,
        start=1
    ):
        print(f"{rank}. {source}")

    # Metrics

    r1 = recall_at_k(
        retrieved_sources,
        expected_sources,
        k=1
    )

    r3 = recall_at_k(
        retrieved_sources,
        expected_sources,
        k=3
    )

    rr = reciprocal_rank(
        retrieved_sources,
        expected_sources
    )

    recall_at_1_scores.append(r1)
    recall_at_3_scores.append(r3)
    mrr_scores.append(rr)

    print(f"\nRecall@1: {r1:.2f}")
    print(f"Recall@3: {r3:.2f}")
    print(f"Reciprocal Rank: {rr:.2f}")


# --------------------------------------------------
# Final evaluation
# --------------------------------------------------

num_questions = len(evaluation_data)

recall_at_1 = (
    sum(recall_at_1_scores)
    / num_questions
)

recall_at_3 = (
    sum(recall_at_3_scores)
    / num_questions
)

mrr_avg = (
    sum(mrr_scores)
    / num_questions
)


print("\n")
print("=" * 60)
print("FINAL RETRIEVAL EVALUATION")
print("=" * 60)

print(f"Questions:  {num_questions}")
print(f"Recall@1:   {recall_at_1:.3f}")
print(f"Recall@3:   {recall_at_3:.3f}")
print(f"MRR:        {mrr_avg:.3f}")