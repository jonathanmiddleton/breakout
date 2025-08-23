import traceback
from openai.types.chat import ChatCompletionToolParam
from tools.tool_protocol import ToolProtocol

class ExecutePython(ToolProtocol):
    _definition = ChatCompletionToolParam(
        type = "function",
        function = {
            "name": "execute_python_script",
            "description": "A complete Python script with a **main** function that prints output. The output "
                           "will be supplied in context for fulfilling the user request.",
            "parameters": {
                "type": "object",
                "properties": {
                    "script": {
                        "type": "string",
                        "description": "The Python script to execute."
                    }
                },
                "required": ["script"]
            }
        }
    )


    @staticmethod
    def _execute_python_script(args: dict) -> None:
        namespace: dict = {}
        try:
            compiled = compile(args["script"], "tool-script", "exec")
            exec(compiled, namespace, namespace)
            if "main" not in namespace or not callable(namespace["main"]):
                raise ValueError("Script must define a callable main() function.")
            namespace["main"]()
        except Exception as e:
            print(f"Error executing Python script: {e}")
            print(traceback.format_exc())

    @staticmethod
    def tool() -> ChatCompletionToolParam:
        return ExecutePython._definition

    DISPATCH = {"execute_python_script": _execute_python_script}