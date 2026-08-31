import asyncio

from app.services.retrieval_service import RetrievalService


async def main():
    service = RetrievalService()

    query = "what are the features of spring boot?"

    print(f"\nQUERY: {query}\n")

    nodes = await service.retrieve_parent_context(query)

    print(f"\nRETURNED NODES: {len(nodes)}")

    for i, node_with_score in enumerate(nodes, 1):
        node = node_with_score.node

        print("\n" + "=" * 80)
        print(f"RESULT {i}")
        print(f"Score: {node_with_score.score}")
        print(f"File: {node.metadata.get('file_name')}")
        print(f"Path: {node.metadata.get('file_path')}")
        print("-" * 80)
        print(node.get_content()[:1500])


if __name__ == "__main__":
    asyncio.run(main())