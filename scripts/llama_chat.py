from llama_index.core import (SimpleDirectoryReader, VectorStoreIndex)
from llama_index.core.schema import NodeWithScore
from llama_index.core.node_parser import HierarchicalNodeParser
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.core.response_synthesizers import get_response_synthesizer

from openai import AsyncOpenAI

from app.core.config import settings
import app.ai.llama

documents = SimpleDirectoryReader("data", recursive=True, required_exts=[".md"]).load_data()
print(f"Documents: {len(documents)}")

#Hierarchical chunking
node_parser = HierarchicalNodeParser.from_defaults(chunk_sizes=[500, 100])
nodes = node_parser.get_nodes_from_documents(documents)

leaf_nodes = [
    node
    for node in nodes
    if node.child_nodes is None
]

print(f"All nodes: {len(nodes)}")
print(f"Leaf nodes: {len(leaf_nodes)}")

docstore = SimpleDocumentStore()
docstore.add_documents(nodes)

#Vector index
index = VectorStoreIndex(leaf_nodes)

#retriever
retriever = index.as_retriever(similarity_top_k=3)

#Response synthesizer
response_synthesizer = get_response_synthesizer(response_mode="compact")

#Groq client
client = AsyncOpenAI(
    api_key=settings.GROQ_API_KEY,
    base_url="https://api.groq.com/openai/v1"
)

MODEL = "llama-3.3-70b-versatile"

#Convo history
conversation = []

#Followup ques
async def contextualize_question(question: str) -> str:
    if not conversation:
        return question
    
    history = "\n".join(
        f"{message['role']}: {message['content']}"
        for message in conversation
    )

    prompt = f"""
Given the conversation below, rewrite the user's latest question
into a standalone question that can be understood without the
conversation history.

Do not answer the question.

Conversation:
{history}

Latest question:
{question}

Standalone question:
"""

    reponse = await client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0
    )

    return reponse.choices[0].message.content.strip()

#Retrieve parent context
def retrieve_parent_context(query: str):
    results = retriever.retrieve(query)

    parent_nodes = {}

    for result in results:
        node = result.node
        if node.parent_node:
            parent_id = node.parent_node.node_id
            parent = docstore.get_document(parent_id)
            parent_nodes[parent_id] = parent
        else:
            parent_nodes[node.node_id] = node
    
    return [
        NodeWithScore(node=node)
        for node in parent_nodes.values()
    ]


def generate_answer(question: str, context_nodes: list[NodeWithScore]):
    response = response_synthesizer.synthesize(question, nodes=context_nodes)
    return str(response)

#Chat loop
async def main():
    print("\n====================================")
    print("   FastAPI Documentation Chatbot")
    print("====================================")
    print("Ask questions about the documentation.")
    print("Type 'exit' to quit.\n")

    while True:
        question = input("You: ").strip()

        if question.lower() == "exit":
            break

        if not question:
            continue

        #Rewrite question using convo
        standalone_question = await contextualize_question(question)
        print(f"\nRetrieval query: {standalone_question}")

        #Retrieve relevant parent context
        context_nodes = retrieve_parent_context(standalone_question)

        #Generate grounded ans
        answer = generate_answer(standalone_question, context_nodes)
        print(f"\nAssistant: {answer}\n")

        #Store convo
        conversation.append(
            {
                "role": "user",
                "content": question,
            }
        )

        conversation.append(
            {
                "role": "assistant",
                "content": answer,
            }
        )

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())


        