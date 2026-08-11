from llama_index.core.schema import BaseNode, TransformComponent

class CharacterCountTransformation(TransformComponent):

    def __call__(self, nodes, **kwargs):
        for node in nodes:
            node.metadata["character_count"] = len(node.text)
            
        return nodes
