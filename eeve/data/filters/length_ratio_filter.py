from typing import Literal

from datatrove.data import Document
from datatrove.pipeline.filters.base_filter import BaseFilter
from datatrove.pipeline.writers.disk_base import DiskWriter

from eeve.utils.datatrove import _get_value


class LengthRatioFilter(BaseFilter):
    name = "📏 Length Ratio"

    def __init__(
        self,
        list_path: list[str],
        ratio_threshold: float,
        exclusion_writer: DiskWriter = None,
        calc_method: Literal["chars", "words"] = "chars",
    ):
        """
        filters if the length ratio between two document fields exceeds the threshold

        Args:
           list_path: two fields in the document for which the filter rule will be checked
           ratio_threshold: maximum allowed ratio between text lengths
           exclusion_writer: writer for excluded documents
           calc_method: unit for length calculation - "chars" for characters, "words" for words (split by spaces)
        """
        super().__init__(exclusion_writer)
        if len(list_path) != 2:
            raise ValueError(
                f"list_path must contain exactly 2 paths, got {len(list_path)}"
            )

        self.list_path = list_path
        self.ratio_threshold = ratio_threshold
        self.calc_method = calc_method

    def filter(self, doc: Document) -> bool | tuple[bool, str]:
        text1, text2 = _get_value(doc, self.list_path)
        if text1 is None or text2 is None:
            return False, "missing_text"

        if self.calc_method == "words":
            source_len = len(text1.split())
            target_len = len(text2.split())
        else:
            source_len = len(text1)
            target_len = len(text2)

        if min(source_len, target_len) == 0:
            return False, "empty_text"

        return (
            max(source_len, target_len) / min(source_len, target_len)
        ) <= self.ratio_threshold
