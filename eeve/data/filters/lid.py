from abc import abstractmethod
from typing import Callable

from datatrove.data import Document
from datatrove.io import cached_asset_path_or_download
from datatrove.pipeline.filters.base_filter import BaseFilter
from datatrove.pipeline.writers.disk_base import DiskWriter
from datatrove.utils._import_utils import check_required_dependencies


class LID:
    def __init__(self, languages: list[str] | None = None) -> None:
        self.languages = languages

    @abstractmethod
    def predict(self, text: str) -> tuple[tuple[str, int], dict[str, float]]:
        """
        Predicts the likelihood of the document being written in given languages, alongside with the most likely language
        Args:
            text (str): Text to predict languages for
        Returns:
            dict[str, float]: Languages and score
        """
        raise NotImplementedError


class FastTextLID(LID):
    MODEL_URL = None
    MODEL_SUBFOLDER = None

    def __init__(
        self,
        model_download_url: str,
        model_subfolder: str,
        languages: list[str] | None = None,
        k: int = -1,
    ) -> None:
        """
        Args:
            languages (list[str]): Languages to predict
            k (int, optional): Number of top-k languages to consider, all languages outside of k will be considered as being predicted with 0.0
        """
        super().__init__(languages)
        self.MODEL_URL = model_download_url
        self.MODEL_SUBFOLDER = model_subfolder
        self._model = None
        self.k = k

    @property
    def model(self):
        if self._model is None:
            check_required_dependencies("lid", [("fasttext", "fasttext-numpy2-wheel")])
            from fasttext.FastText import _FastText

            model_file = cached_asset_path_or_download(
                self.MODEL_URL,
                namespace="lid",
                subfolder=self.MODEL_SUBFOLDER,
                desc="fast-text language identifier model",
            )
            self._model = _FastText(model_file)
        return self._model

    def predict(self, text: str) -> tuple[tuple[str, int], dict[str, float]]:
        langs, scores = self.model.predict(text.replace("\n", " "), k=self.k)
        lang_pairs = {
            lang.split("__")[2]: score.item() for lang, score in zip(langs, scores)
        }
        best_lang_pair = max(lang_pairs.items(), key=lambda x: x[1])
        return best_lang_pair, {
            lang: lang_pairs.get(lang, 0.0) for lang in self.languages
        } if self.languages else lang_pairs


# adapted from https://github.com/huggingface/datatrove/blob/main/src/datatrove/pipeline/filters/language_filter.py
class LanguageFilter(BaseFilter):
    name = "🌍 Language ID"
    _requires_dependencies = [("fasttext", "fasttext-numpy2-wheel"), "fasteners"]

    def __init__(
        self,
        model_download_url: str,
        model_subfolder: str = "fasttext_model",
        languages: list[str] | str | None = None,
        language_threshold: float = 0.65,
        exclusion_writer: DiskWriter = None,
        label_only: bool = False,
        keep_top_pairs_threshold: float = -1,
        content_extractor: Callable[[Document], str] = lambda doc: doc.text,
    ):
        """
        filters if the predicted language is not among given language or if the language score is below language
        language_threshold

        Args:
            languages: list of languages to keep. None for all
            language_threshold: language_threshold minimum score to accept a document
            exclusion_writer: writer for saving intermediate execution results
            label_only: if True, only the language label is added to the metadata and no documents are removed
            keep_top_pairs_threshold: keep a list of all language pairs with at least this score. -1 to disable
            content_extractor: function for extracting data from an object Document (initially only filtering of doc.text is supported)
        """
        super().__init__(exclusion_writer)
        self.language_threshold = language_threshold
        if isinstance(languages, str):
            languages = [languages]
        self.languages = languages
        self.model = FastTextLID(
            model_download_url=model_download_url,
            model_subfolder=model_subfolder,
            languages=languages,
        )
        self.label_only = label_only
        self.keep_top_pairs_threshold = keep_top_pairs_threshold
        self.content_extractor = content_extractor

    def filter(self, doc: Document) -> bool:
        content = self.content_extractor(doc)
        best_lang_pair, lang_pairs = self.model.predict(content)
        lang, lang_score = best_lang_pair
        doc.metadata["language"] = lang
        doc.metadata["language_score"] = lang_score
        if self.keep_top_pairs_threshold != -1:
            for key, value in lang_pairs.items():
                if value > self.keep_top_pairs_threshold:
                    doc.metadata[f"top_language_{key}_score"] = value
        return (
            self.label_only
            or (
                self.languages
                and any(
                    score > self.language_threshold for score in lang_pairs.values()
                )
            )
            or (self.languages is None and lang_score > self.language_threshold)
        )
