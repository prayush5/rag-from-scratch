from llama_index.core import Settings
from llama_index.core.llms import ChatMessage, MessageRole
from llama_index.core.schema import NodeWithScore
from llama_index.core.response_synthesizers import get_response_synthesizer
from app.core.prompts import RAG_SYSTEM_PROMPT
from app.core.exceptions import GenerationError
from app.observability.langfuse import langfuse
from typing import AsyncGenerator

class GenerationService:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.response_synthesizer = get_response_synthesizer(
            response_mode="compact",
            text_qa_template=RAG_SYSTEM_PROMPT,
            streaming=True
        )

    async def stream_response(self, standalone_question: str, context_nodes: list[NodeWithScore]) -> AsyncGenerator[str, None]:
        context_str = "\n\n".join(node.node.get_content() for node in context_nodes)

        prompt = RAG_SYSTEM_PROMPT.format(
            context_str=context_str,
            query_str=standalone_question,
        )

        with langfuse.start_as_current_observation(
            name="response-generation-stream",
            as_type="generation",
            input={"prompt_query": standalone_question, "context_snippets": [n.node.get_content() for n in context_nodes]},
            model=self.model_name
        ) as gen_obs:
            try:
                accumulated_text = []
                response_stream = await Settings.llm.astream_complete(prompt)

                async for chunk in response_stream:
                    delta = chunk.delta
                    if delta:
                        accumulated_text.append(delta)
                        yield delta

                gen_obs.update(output="".join(accumulated_text))

            except Exception as ex:
                gen_obs.update(level="ERROR", status_message=str(ex))
                raise GenerationError("Failed during response streaming") from ex
    
    async def generate_response(self, standalone_question: str, context_nodes: list[NodeWithScore]) -> str:
        with langfuse.start_as_current_observation(
            name="response-generation",
            as_type="generation",
            input={
                "prompt_query": standalone_question,
                "context_snippets": [node.node.get_content() for node in context_nodes],
            },
            model=self.model_name,
        ) as gen_obs:
            try:
                raw_response = await self.response_synthesizer.asynthesize(
                    standalone_question, 
                    nodes=context_nodes
                )
                answer = str(raw_response)
                gen_obs.update(output=answer)
                return answer
            except Exception as ex:
                gen_obs.update(level="ERROR", status_message=str(ex))
                raise GenerationError("Failed to generate response") from ex

