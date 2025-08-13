import re
from collections import Counter


def compare_length(
    source_text: str,
    target_text: str,
    max_length_ratio: float
) -> bool:
    source_len = len(source_text)
    target_len = len(target_text)

    if min(source_len, target_len) == 0:
        return source_len == target_len
    return (max(source_len, target_len) / min(source_len, target_len)) <= max_length_ratio


def regex_parallel_corpus_compare(source_sentence: str, target_sentence: str, regex_expression) -> bool:
    def count_smth(text: str, regex):
        matches = re.findall(regex, text)
        return Counter(matches)
    return count_smth(source_sentence, regex_expression) == count_smth(target_sentence, regex_expression)
