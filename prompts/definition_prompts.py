from openai.types.chat import (ChatCompletionUserMessageParam,
                               ChatCompletionSystemMessageParam,
                               ChatCompletionDeveloperMessageParam)

_SYSTEM_PROMPT = """
            You are a helpful dictionary assistant. Use tools to gather more information if necessary. 
            If a tool supplies new unknown terms then use the tool for the unknown term. Continue to 
            repeatedly use the tool on unknown terms until you can supply a definition using only common, 
            everyday words.
"""


def get_messages(p: str) -> list[ChatCompletionSystemMessageParam
                                  | ChatCompletionUserMessageParam
                                  | ChatCompletionDeveloperMessageParam]:
    messages: list[ChatCompletionSystemMessageParam | ChatCompletionUserMessageParam
                   | ChatCompletionDeveloperMessageParam ] = [
        ChatCompletionSystemMessageParam(role="system", content=_SYSTEM_PROMPT),
        ChatCompletionUserMessageParam(role="user", content=p)]

    return messages
