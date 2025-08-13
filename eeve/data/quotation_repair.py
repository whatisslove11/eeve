"""
Function for Standardizing Quotation Marks

Standardizes quotation marks in text by replacing them with chevron quotes (« »).
In case of nested quotation marks, only replaces the outermost pair with chevron quotes, leaving the inner ones unchanged.
"""
import re
from abc import ABC, abstractmethod
from typing import Any

from abc import ABC, abstractmethod
from datatrove.data import DocumentsPipeline
from datatrove.pipeline.base import PipelineStep
from datatrove.utils.typeshelper import StatHints


class AdvancedFormatter(PipelineStep, ABC):
    type = "✂️ - FORMAT"

    def __init__(self, field_path: str = "text"):
        super().__init__()
        self.field_path = field_path
        
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
                current_value = self._get_value(doc, self.field_path)
                formatted_value = self.format(current_value)
                self._set_value(doc, self.field_path, formatted_value)
            yield doc


class QuoteReplacer(AdvancedFormatter):
    name = "Quotation Mark Replacer"

    def __init__(self, field_path: str = "text"):
        super().__init__(field_path=field_path)

    def format(self, text: str) -> str:
        if not text:
            return text
        
        result = []
        quote_stack = []  
        i = 0
        
        while i < len(text):
            if text[i] == '"':
                if not quote_stack:
                    result.append('«')
                    quote_stack.append('guillemet')
                elif quote_stack[-1] == 'guillemet':
                    if i + 1 < len(text) and text[i + 1].isalpha():
                        result.append('"')
                        quote_stack.append('quote')
                    else:
                        result.append('»')
                        quote_stack.pop()
                elif quote_stack[-1] == 'quote':
                    result.append('"')
                    quote_stack.pop()
            else:
                result.append(text[i])
            i += 1
        
        return ''.join(result)