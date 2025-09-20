import re
from collections import Counter
from datatrove.data import Document
from datatrove.pipeline.filters.base_filter import BaseFilter
from datatrove.pipeline.writers.disk_base import DiskWriter
from eeve.utils.datatrove import _get_value


class RegexCounterFilter(BaseFilter):
    name = "🧮 Regex Counter"

    def __init__(self, list_path: list[str], regex_expr, exclusion_writer: DiskWriter = None):
        super().__init__(exclusion_writer)
        if len(list_path) != 2:
            raise ValueError
        
        self.list_path = list_path
        self.regex = re.compile(regex_expr)

    def filter(self, doc: Document) -> bool | tuple[bool, str]:
        text1, text2 = _get_value(doc, self.list_path)
        if text1 is None or text2 is None:
            return False, "missing_text"
        
        return Counter(self.regex.findall(text1)) == Counter(self.regex.findall(text2))