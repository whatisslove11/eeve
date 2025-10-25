from typing import Callable

from eeve.data.formatters.base import AdvancedFormatter


class CallableFormatter(AdvancedFormatter):
    name = "🔧 Custom"

    def __init__(self, func: Callable[[str], str], list_path: str | list[str] = "text"):
        super().__init__(list_path=list_path)
        self.func = func

    def format(self, text) -> str:
        return self.func(text)
