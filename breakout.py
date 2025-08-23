from tools.tool_registry import ToolRegistry
from tools.python_tool import ExecutePython
from prompts import get_messages
import json
from openai import OpenAI

registry = ToolRegistry()
registry.register_tool(ExecutePython.tool(), ExecutePython.DISPATCH)

def _completion(query: str) -> None:
    base_url = "http://localhost:8080/v1"
    client = OpenAI(base_url=base_url, api_key="")
    with client.chat.completions.stream(
            messages=get_messages(query,registry.tools()),
            temperature=0.6,
            tools=registry.tools(),
            model="",
            top_p=0.95,
            frequency_penalty=1.1,
            seed=1337,
            max_tokens=4096,
    ) as stream:
        for event in stream:
            if event.type == "tool_calls.function.arguments.done":
                name, args = event.name, json.loads(event.arguments)
                registry.execute_tool(name, args)
            elif event.type in ["tool_calls.function.arguments.delta"]:
                continue
            elif event.type == "chunk":
                try:
                    reasoning_content = event.chunk.choices[0].delta.model_extra.get("reasoning_content")
                    if reasoning_content is not None:
                        print(reasoning_content, end="", flush=True)
                except Exception as e:
                    print(f"Error: {e}")
            elif event.type == "content.delta":
                print(event.delta, end="", flush=True)
            elif event.type == "content.done":
                print(event.content, end="", flush=True)
            else:
                print(f"Event: {event}")

if __name__ == "__main__":
    _completion("Tell me the weather in London without asking me any questions.")