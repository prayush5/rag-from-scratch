from scripts.llama_chat import (
    retriever,
    retrieve_bm25,
    reciprocal_rank_fusion,
    leaf_nodes,
)


def main():

    query = "how does dependency injection enforce DRY?"

    # Dense retrieval
    dense_results = retriever.retrieve(query)

    dense_indices = [
        leaf_nodes.index(result.node)
        for result in dense_results
    ]

    # BM25 retrieval
    bm25_indices = retrieve_bm25(query)

    # Fusion
    fused_indices = reciprocal_rank_fusion(
        dense_indices,
        bm25_indices
    )

    print("\nDense:")
    for index in dense_indices[:5]:
        print(
            index,
            leaf_nodes[index].metadata.get("file_name")
        )

    print("\nBM25:")
    for index in bm25_indices[:5]:
        print(
            index,
            leaf_nodes[index].metadata.get("file_name")
        )

    print("\nFused:")
    for rank, index in enumerate(
        fused_indices[:10],
        start=1
    ):
        print(
            rank,
            index,
            leaf_nodes[index].metadata.get("file_name")
        )


if __name__ == "__main__":
    main()