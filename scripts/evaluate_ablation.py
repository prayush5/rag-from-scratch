import json

from llama_index.core import (
    SimpleDirectoryReader,
    VectorStoreIndex,
    StorageContext,
)
from llama_index.core.node_parser import HierarchicalNodeParser
from llama_index.core.retrievers import QueryFusionRetriever, AutoMergingRetriever
from llama_index.retrievers.bm25 import BM25Retriever

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
# Storage context
# --------------------------------------------------

from llama_index.core.storage.docstore import SimpleDocumentStore

docstore = SimpleDocumentStore()
docstore.add_documents(nodes)

storage_context = StorageContext.from_defaults(
    docstore=docstore
)


# --------------------------------------------------
# Dense retriever
# --------------------------------------------------

index = VectorStoreIndex(
    leaf_nodes,
    storage_context=storage_context
)

dense_retriever = index.as_retriever(
    similarity_top_k=10
)


# --------------------------------------------------
# BM25 retriever
# --------------------------------------------------

bm25_docstore = SimpleDocumentStore()
bm25_docstore.add_documents(leaf_nodes)

bm25_retriever = BM25Retriever.from_defaults(
    docstore=bm25_docstore,
    similarity_top_k=10
)


# --------------------------------------------------
# Hybrid retriever
# --------------------------------------------------

hybrid_retriever = QueryFusionRetriever(
    retrievers=[
        dense_retriever,
        bm25_retriever,
    ],
    similarity_top_k=10,
    num_queries=1,
    mode="reciprocal_rerank",
    use_async=False,
)


# --------------------------------------------------
# Auto-merging retriever
# --------------------------------------------------

auto_merging_retriever = AutoMergingRetriever(
    vector_retriever=dense_retriever,
    storage_context=storage_context,
    simple_ratio_thresh=0.5,
)


# --------------------------------------------------
# Helper
# --------------------------------------------------

def unique_sources(results) -> list[str]:

    sources = []
    seen = set()

    for result in results:

        source = result.node.metadata.get(
            "file_name"
        )

        if source and source not in seen:
            sources.append(source)
            seen.add(source)

    return sources


# --------------------------------------------------
# Metrics
# --------------------------------------------------

def recall_at_k(
    retrieved_sources: list[str],
    expected_sources: set[str],
    k: int,
) -> float:

    retrieved = set(
        retrieved_sources[:k]
    )

    return float(
        bool(retrieved & expected_sources)
    )


def reciprocal_rank(
    retrieved_sources: list[str],
    expected_sources: set[str],
) -> float:

    for rank, source in enumerate(
        retrieved_sources,
        start=1,
    ):

        if source in expected_sources:
            return 1 / rank

    return 0.0


# --------------------------------------------------
# Evaluation function
# --------------------------------------------------

def evaluate_retriever(
    name: str,
    retriever,
):

    recall_at_1_scores = []
    recall_at_3_scores = []
    mrr_scores = []

    print("\n")
    print("=" * 60)
    print(f" {name}")
    print("=" * 60)

    for item in evaluation_data:

        question = item["question"]

        expected_sources = set(
            item["expected_sources"]
        )

        results = retriever.retrieve(
            question
        )

        retrieved_sources = unique_sources(
            results
        )

        r1 = recall_at_k(
            retrieved_sources,
            expected_sources,
            k=1,
        )

        r3 = recall_at_k(
            retrieved_sources,
            expected_sources,
            k=3,
        )

        rr = reciprocal_rank(
            retrieved_sources,
            expected_sources,
        )

        recall_at_1_scores.append(r1)
        recall_at_3_scores.append(r3)
        mrr_scores.append(rr)

        print(
            f"{question[:45]:45} "
            f"R@1={r1:.2f} "
            f"R@3={r3:.2f} "
            f"MRR={rr:.2f}"
        )

    num_questions = len(
        evaluation_data
    )

    recall_at_1 = (
        sum(recall_at_1_scores)
        / num_questions
    )

    recall_at_3 = (
        sum(recall_at_3_scores)
        / num_questions
    )

    mrr = (
        sum(mrr_scores)
        / num_questions
    )

    print("\nResults:")
    print(
        f"Recall@1: {recall_at_1:.3f}"
    )
    print(
        f"Recall@3: {recall_at_3:.3f}"
    )
    print(
        f"MRR:      {mrr:.3f}"
    )

    return {
        "Recall@1": recall_at_1,
        "Recall@3": recall_at_3,
        "MRR": mrr,
    }


# --------------------------------------------------
# Run experiments
# --------------------------------------------------

results = {}


results["Dense"] = evaluate_retriever(
    "Dense Retrieval",
    dense_retriever,
)


results["Hybrid"] = evaluate_retriever(
    "Hybrid Retrieval (Dense + BM25)",
    hybrid_retriever,
)


results["Auto-Merging"] = evaluate_retriever(
    "Auto-Merging Retrieval",
    auto_merging_retriever,
)


# --------------------------------------------------
# Final comparison
# --------------------------------------------------

print("\n")
print("=" * 75)
print("ABLATION RESULTS")
print("=" * 75)

print(
    f"{'Pipeline':30}"
    f"{'Recall@1':>12}"
    f"{'Recall@3':>12}"
    f"{'MRR':>12}"
)

print("-" * 75)

for name, metrics in results.items():

    print(
        f"{name:30}"
        f"{metrics['Recall@1']:>12.3f}"
        f"{metrics['Recall@3']:>12.3f}"
        f"{metrics['MRR']:>12.3f}"
    )