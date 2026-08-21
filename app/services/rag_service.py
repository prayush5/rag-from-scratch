from openai import AsyncOpenAI
from llama_index.core.postprocessor import LongContextReorder
from typing import AsyncGenerator, Any
from fastapi.encoders import jsonable_encoder

from app.core.config import settings
from app.ai.context_selector import select_context
from app.ai.query_rewriter import QueryRewriter
from app.schemas.chat import Message, ChatResponse
from app.services.guardrail_service import GuardRailService
from app.services.retrieval_service import RetrievalService
from app.services.generation_service import GenerationService
from app.observability.langfuse import langfuse

class RAGService:
    def __init__(self):
        self.model = "openai/gpt-oss-120b"
        self.llm_client = AsyncOpenAI(
            api_key=settings.GROQ_API_KEY,
            base_url="https://api.groq.com/openai/v1"
        )

        self.guardrail = GuardRailService()
        self.retrieval_service = RetrievalService()
        self.generation_service = GenerationService(model_name=self.model)
        self.query_rewriter = QueryRewriter(llm_client=self.llm_client, model=self.model)
        self.context_reorder = LongContextReorder()


    async def stream_answer_question(
        self, question: str, history: list[Message]
    ) -> AsyncGenerator[tuple[str, Any], None]:
        # 1. Guardrails, rewriting, and context retrieval
        await self.guardrail.validate_content(text=question, role="user")
        standalone_question = await self.query_rewriter.rewrite(question, history)
        retrieved_nodes = await self.retrieval_service.retrieve_parent_context(standalone_question)
        selected_nodes = select_context(retrieved_nodes, max_tokens=2000)
        context_nodes = await self.context_reorder.apostprocess_nodes(selected_nodes)

        sources = [
            node.node.metadata.get("file_name") 
            for node in context_nodes 
            if node.node.metadata.get("file_name")
        ]

        async for chunk in self.generation_service.stream_response(standalone_question, context_nodes):
            yield ("token", chunk) 

        yield ("metadata", {"sources": list(set(sources)), "standalone_query": standalone_question})

    async def answer_question(self, question: str, history: list[Message]) -> ChatResponse:
        with langfuse.start_as_current_observation(
            name="rag-chat",
            as_type="chain",
            input={
                "question": question,
                "history": [msg.model_dump() for msg in history]
            }
        ) as trace:
            #1. input guardrail
            with langfuse.start_as_current_observation(
                name="input-guardrail",
                as_type="guardrail",
                input={
                    "text": question,
                    "role": "user"
                }
            ) as guard_obs:
                
                await self.guardrail.validate_content(text=question, role="user")
                guard_obs.update(output={"status": "safe"})

            #2. query rewriting
            standalone_question = await self.query_rewriter.rewrite(question, history)

            #3. retrieval
            retrieved_parent_nodes = await self.retrieval_service.retrieve_parent_context(standalone_question)

            #4. context selection and reordering
            with langfuse.start_as_current_observation(
                name="context-selection",
                as_type="chain",
                input={
                    "input_nodes": len(retrieved_parent_nodes),
                    "max_tokens": 2000
                }
            ) as context_obs:
                selected_nodes = select_context(retrieved_parent_nodes, max_tokens=2000)
                estimated_tokens = sum(len(node.node.get_content()) // 4 for node in selected_nodes)

                context_obs.update(
                    output={
                        "selected_nodes": len(selected_nodes),
                        "estimated_tokens": estimated_tokens
                    }
                )

            context_nodes = await self.context_reorder.apostprocess_nodes(selected_nodes)

            #5. answer generation
            answer = await self.generation_service.generate_response(standalone_question, context_nodes)

            #6. extract sources
            context = "\n\n".join(node.node.get_content() for node in context_nodes)
            sources = []
            for node in context_nodes:
                source = node.node.metadata.get("file_name")
                if source and source not in sources:
                    sources.append(source)
            
            #7. output guardrail
            with langfuse.start_as_current_observation(
                name="output-guardrail",
                as_type="guardrail",
                input={
                    "text": answer,
                    "role": "agent"
                }
            ) as out_guard_obs:
                await self.guardrail.validate_content(text=answer, role="agent")
                out_guard_obs.update(output={"status": "safe"})
        
            trace.update(
                output={
                    "standalone_query": standalone_question,
                    "answer": answer,
                    "sources": sources
                }
            )
        
            return ChatResponse(
                question=question,
                standalone_query=standalone_question,
                answer=answer,
                sources=sources,
                context=context,
            )