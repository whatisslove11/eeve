from abc import ABC, abstractmethod
from datatrove.data import DocumentsPipeline
from datatrove.pipeline.base import PipelineStep
from datatrove.utils.typeshelper import StatHints


class AdvancedFormatter(PipelineStep, ABC):
    """
    An enhanced version of BaseFormatter. Allows working not only with the "text" field of the Document class, but with any other field as well.
    To access the desired field, simply pass a string with the description of the field you want to access.
    Examples:
    - "metadata['abrakadabra']"
    - "text" (set by default)
    """
    type = "✂️ - FORMAT"

    def __init__(self, list_path: str | list[str]):
        super().__init__()
        self.list_path = [list_path] if isinstance(list_path, str) else list_path
        
    def _get_value(self, doc, path: str) -> str:
        context = {"doc": doc}
        return eval(f"doc.{path}", context)
    
    def _set_value(self, doc, path: str, value: str):
        if "[" in path:
            base_path, key_part = path.split("[", 1)
            key = key_part.rstrip("]").strip("'\"")
            base_obj = getattr(doc, base_path)
            base_obj[key] = value
        else:
            setattr(doc, path, value)

    @abstractmethod
    def format(self, text: str) -> str:
        return text

    def run(self, data: DocumentsPipeline, rank: int = 0, world_size: int = 1) -> DocumentsPipeline:
        for doc in data:
            self.stat_update(StatHints.total)
            with self.track_time():
                for field_path in self.list_path:
                    current_value = self._get_value(doc, field_path)
                    formatted_value = self.format(current_value)
                    self._set_value(doc, field_path, formatted_value)
            yield doc