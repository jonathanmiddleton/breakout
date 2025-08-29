from openai.types.chat import ChatCompletionToolParam
from tools.tool_protocol import ToolProtocol

class DefinitionTool(ToolProtocol):
    _definition = ChatCompletionToolParam(
        type = "function",
        function = {
            "name": "get_definition",
            "description": "Retrieves the defintion of an unknown term.",
            "parameters": {
                "type": "object",
                "properties": {
                    "term": {
                        "type": "string",
                        "description": "The term to define."
                    }
                },
                "required": ["term"]
            }
        }
    )

    TERMS = {
        "solidgoldmagikarp" : "See the definition of 'swugglefuster'.",
        "swugglefuster" : "See the definition of 'chocoblockoswizzlefuster'.",
        "chocoblockoswizzlefuster" : "A chocolate block made from dogs.",
        ##### Circular References #####
        "solidsilvermagikorp" : "See the definition of 'swigglefister'.",
        "swigglefister" : "See the definition of 'sporkyporkydorky'.",
        "sporkyporkydorky" : "See the definition of 'solidsilvermagikorp'.",
    }

    @staticmethod
    def _get_definition(args: dict) -> tuple[str, str]:
        term = args["term"].lower()
        return term, DefinitionTool.TERMS[term]


    @staticmethod
    def tool() -> ChatCompletionToolParam:
        return DefinitionTool._definition

    DISPATCH = {"get_definition": _get_definition}