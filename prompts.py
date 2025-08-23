from openai.types.chat import ChatCompletionFunctionToolParam, ChatCompletionUserMessageParam, \
    ChatCompletionSystemMessageParam

_SYSTEM_PROMPT = """
            You are a helpful assistant. Think step by step and evaluate whether the 
            provided context is sufficient to answer a user's query **accurately**. 
            If the supplied context is insufficient, use a tool to gather more information. 
            You are allowed to write Python scripts to fulfill the user's request.
"""

def get_messages(p: str, tools: list[ChatCompletionFunctionToolParam] = None) -> list[ChatCompletionSystemMessageParam | ChatCompletionUserMessageParam]:
    messages: list[ChatCompletionSystemMessageParam | ChatCompletionUserMessageParam] = [ChatCompletionUserMessageParam(role="user", content=p)]
    if tools:
        ts = [str(t) for t in tools]
        sp = (_SYSTEM_PROMPT + "You have access to the following tools: " +  "".join(ts))
        messages = [ChatCompletionSystemMessageParam(role="system", content=sp)] + messages
    else:
        messages = [ChatCompletionSystemMessageParam(role="system", content=_SYSTEM_PROMPT)] + messages
    return messages