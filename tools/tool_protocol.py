from typing import Protocol

class ToolProtocol(Protocol):
    @staticmethod
    def tool():
        ...

    @staticmethod
    def _get_params(_tool: dict) -> dict:
        return _tool["function"]["parameters"]["properties"]