from tools.definition_tool import DefinitionTool
from tools.tool_registry import ToolRegistry
from openai import OpenAI
from openai_harmony import (
    Author,
    Conversation,
    DeveloperContent,
    HarmonyEncodingName,
    Message,
    Role,
    SystemContent,
    ToolDescription,
    load_harmony_encoding,
    ReasoningEffort
)
import json
from typing import Any

encoding = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)

system_message = (
    SystemContent.new()
    .with_reasoning_effort(ReasoningEffort.HIGH)
    .with_conversation_start_date("2025-06-28")
)

developer_message = (
    DeveloperContent.new()
    .with_instructions("Always respond in riddles")
    .with_function_tools(
        [
            ToolDescription.new(
                "get_current_weather",
                "Gets the current weather in the provided location.",
                parameters={
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "The city and state, e.g. San Francisco, CA",
                        },
                        "format": {
                            "type": "string",
                            "enum": ["celsius", "fahrenheit"],
                            "default": "celsius",
                        },
                    },
                    "required": ["location"],
                },
            ),
        ]
    )
)

convo = Conversation.from_messages(
    [
        Message.from_role_and_content(Role.SYSTEM, system_message),
        Message.from_role_and_content(Role.DEVELOPER, developer_message),
        Message.from_role_and_content(Role.USER, "What is the weather in Tokyo?"),
        # Message.from_role_and_content(
        #     Role.ASSISTANT,
        #     'User asks: "What is the weather in Tokyo?" We need to use get_current_weather tool.',
        # ).with_channel("analysis"),
        # Message.from_role_and_content(Role.ASSISTANT, '{"location": "Tokyo"}')
        # .with_channel("commentary")
        # .with_recipient("functions.get_current_weather")
        # .with_content_type("<|constrain|> json"),
        # Message.from_author_and_content(
        #     Author.new(Role.TOOL, "functions.get_current_weather"),
        #     '{ "temperature": 20, "sunny": true }',
        # ).with_channel("commentary"),
    ]
)

tokens = encoding.render_conversation_for_completion(convo, Role.ASSISTANT)


registry = ToolRegistry()
registry.register_tool(DefinitionTool.tool(), DefinitionTool.DISPATCH)

def _completion(messages: list[Any]) -> None:
    base_url = "http://localhost:1234/v1"
    client = OpenAI(base_url=base_url, api_key="")
    with client.chat.completions.stream(
            messages=[Message.from_role_and_content(Role.USER, "What is the weather in Tokyo?")],
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

        # messages.append(ChatCompletionSystemMessageParam(
        #     role="system",
        #     content=f"{{\"The assistant called the tool {name} with the following arguments: {arg} and returned the following result: {result}\"}}")
        # )
        # _completion(messages)

if __name__ == "__main__":
    query = "What is a 'solidgoldmagikarp'?"
    #query = "What is a 'solidsilvermagikorp'?"
    # _completion(get_messages(query))
    _completion([Message.from_role_and_content(Role.USER, query)])