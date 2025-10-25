import re
from collections import Counter

from datatrove.data import Document
from datatrove.pipeline.filters.base_filter import BaseFilter
from datatrove.pipeline.writers.disk_base import DiskWriter

from eeve.utils.datatrove import _get_value


class RegexCounterFilter(BaseFilter):
    name = "🧮 Regex Counter"

    def __init__(
        self, list_path: list[str], regex_expr: str, exclusion_writer: DiskWriter = None
    ):
        """
        filters if the count of regex matches in two document fields are equal

        Args:
           list_path: two fields in the document for which the filter rule will be checked
           regex_expr: regex expression
           exclusion_writer:
        """
        super().__init__(exclusion_writer)
        if len(list_path) != 2:
            raise ValueError(
                f"list_path must contain exactly 2 paths, got {len(list_path)}"
            )

        self.list_path = list_path
        self.regex = re.compile(regex_expr)

    def filter(self, doc: Document) -> bool | tuple[bool, str]:
        text1, text2 = _get_value(doc, self.list_path)
        if text1 is None or text2 is None:
            return False, "missing_text"

        return Counter(self.regex.findall(text1)) == Counter(self.regex.findall(text2))
