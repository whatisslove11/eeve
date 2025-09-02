from datatrove.data import Document
from datatrove.pipeline.filters.base_filter import BaseFilter
from datatrove.pipeline.writers.disk_base import DiskWriter
from eeve.utils.datatrove import _get_value


class LengthRatioFilter(BaseFilter):
    def __init__(self, list_path: list[str], ratio_threshold: float, exclusion_writer: DiskWriter = None):
        super().__init__(exclusion_writer)
        if len(list_path) != 2:
            raise ValueError
        
        self.list_path = list_path
        self.ratio_threshold = ratio_threshold

    def filter(self, doc: Document) -> bool | tuple[bool, str]:
        text1, text2 = _get_value(doc, self.list_path)
        if text1 is None or text2 is None:
            return False, "missing_text"
        
        source_len = len(text1)
        target_len = len(text2)

        if min(source_len, target_len) == 0:
            return False, "empty_text"
        return (max(source_len, target_len) / min(source_len, target_len)) <= self.ratio_threshold