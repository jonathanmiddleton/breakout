from typing import Any

from openai.types.chat import  ChatCompletionSystemMessageParam

from tools.tool_registry import ToolRegistry
from tools.definition_tool import DefinitionTool
from prompts.definition_prompts import get_messages
import json
from openai import OpenAI

registry = ToolRegistry()
registry.register_tool(DefinitionTool.tool(), DefinitionTool.DISPATCH)

def _completion(messages: list[Any]) -> None:
    base_url = "http://localhost:8228/v1"
    client = OpenAI(base_url=base_url, api_key="")
    with client.chat.completions.stream(
            messages=messages,
            temperature=0.0,
            tools=registry.tools(),
            tool_choice="auto",
            model="",
            top_p=0.95,
            frequency_penalty=1.1,
            seed=1337,
            max_tokens=4096,
    ) as stream:
        for event in stream:
            if event.type == "tool_calls.function.arguments.done":
                name, args = event.name, json.loads(event.arguments)
                arg, result = registry.execute_tool(name, args)
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
                return
            else:
                print(f"Event: {event}")

        print("\n\n")
        print(f"argument: {arg}, result: {result}")
        print("\n\n")

        messages.append(ChatCompletionSystemMessageParam(
            role="system",
            content=f"{{\"The assistant called the tool {name} with the following arguments: {arg} and returned the following result: {result}\"}}")
        )
        _completion(messages)

if __name__ == "__main__":
    #query = "What is a 'solidgoldmagikarp'?"
    query = "What is a 'solidsilvermagikorp'?"
    _completion(get_messages(query))