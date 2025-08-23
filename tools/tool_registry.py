from openai.types.chat import ChatCompletionFunctionToolParam


class ToolRegistry:
    _DISPATCH = {}
    _TOOLS: list[ChatCompletionFunctionToolParam] = []

    def register_tool(self, tool: ChatCompletionFunctionToolParam, dispatch: dict) -> None:
        self._DISPATCH.update(dispatch)
        self._TOOLS.append(tool)

    def execute_tool(self, name: str, args: dict):
        return self._DISPATCH[name](args)

    def tools(self) -> list[ChatCompletionFunctionToolParam]:
        return self._TOOLS