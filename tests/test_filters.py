import os
import unittest

import pytest

from eeve.data.filters import (
    BadTranslationsFilter,
    LanguageFilter,
    LengthRatioFilter,
    RegexCounterFilter,
)
from eeve.utils.datatrove import fasttext_model_get_path

from .utils import (
    make_doc,
    require_fasttext,
    require_hf_hub,
    require_openai,
)


FLOAT_BASIC_PATTERN = r"[+-]?\d+(?:\.\d+)?"
FLOAT_EXTENDED_PATTERN = r"[+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?"
TASK_DESC = "Retrieve parallel sentences."

TEXT_EN_PARALLEL = (
    "During the annual audit, we reviewed 3 invoices for the project. The net amount was 17.75 USD, "
    "with tax +2.00 and a discount of -5.75 applied. The sensor’s tolerance was specified as 1e-3; "
    "the storage capacity reached 2.5E+10 bytes. We also added .75 liters of coolant and recorded asset ID 1 "
    "in the register."
)
TEXT_RU_PARALLEL = (
    "Во время ежегодного аудита мы проверили 3 счета по проекту. Чистая сумма составила 17.75 USD, "
    "начислен налог +2.00 и применена скидка -5.75. Допуск датчика указан как 1e-3; "
    "ёмкость хранилища достигала 2.5E+10 байт. Мы также добавили .75 литра охлаждающей жидкости и внесли ID 1 в реестр "
    "идентификатор актива."
)
TEXT_FR_PARALLEL = (
    "Lors de l’audit annuel, nous avons examiné 3 factures du projet. Le montant net était de 17.75 USD, "
    "avec une taxe de +2.00 et une remise de -5.75. La tolérance du capteur était indiquée à 1e-3; "
    "la capacité de stockage atteignait 2.5E+10 octets. Nous avons aussi ajouté .75 litre de liquide de refroidissement "
    "et enregistré l’identifiant d’actif 1 dans le registre."
)
TEXT_FR_PARALLEL_DIFF = TEXT_FR_PARALLEL.replace("17.75", "17.76")


class TestFilters(unittest.TestCase):
    def check_filter(self, filter, doc, filter_reason):
        filter_result = filter.filter(doc)
        self.assertEqual(type(filter_result), tuple)
        self.assertEqual(filter_result[1], filter_reason)

    def test_regex_counter_filter(self):
        counter_basic = RegexCounterFilter(
            list_path=["text", "metadata['test_text']"],
            regex_expr=FLOAT_BASIC_PATTERN,
        )
        counter_extended = RegexCounterFilter(
            list_path=["text", "metadata['test_text']"],
            regex_expr=FLOAT_EXTENDED_PATTERN,
        )

        doc_parallel_en_ru = make_doc(TEXT_EN_PARALLEL, metadata_text=TEXT_RU_PARALLEL)
        self.assertTrue(counter_basic.filter(doc_parallel_en_ru))
        self.assertTrue(counter_extended.filter(doc_parallel_en_ru))

        doc_parallel_en_fr_diff = make_doc(
            TEXT_EN_PARALLEL, metadata_text=TEXT_FR_PARALLEL_DIFF
        )
        self.assertFalse(counter_basic.filter(doc_parallel_en_fr_diff))
        self.assertFalse(counter_extended.filter(doc_parallel_en_fr_diff))

        doc_equal_simple = make_doc(
            "Order 3 units, price -5.75, tax +2.00 and 12 items; total 17.75"
        )
        self.assertTrue(counter_basic.filter(doc_equal_simple))
        self.assertTrue(counter_extended.filter(doc_equal_simple))

        doc_diff_last_digit = make_doc(
            "v1=10 and v2=2.50", metadata_text="v1=10 and v2=2.51"
        )
        self.assertFalse(counter_basic.filter(doc_diff_last_digit))
        self.assertFalse(counter_extended.filter(doc_diff_last_digit))

        doc_equal_with_scientific = make_doc(
            "values: 1e-3 and 2.5E+10; extra: .75 and 1."
        )
        self.assertTrue(counter_basic.filter(doc_equal_with_scientific))
        self.assertTrue(counter_extended.filter(doc_equal_with_scientific))

        doc_scientific_split = make_doc(
            "a=1e-3 b=2.5E+10", metadata_text="a 1 -3 b 2.5 +10"
        )
        self.assertTrue(counter_basic.filter(doc_scientific_split))
        self.assertFalse(counter_extended.filter(doc_scientific_split))

        doc_reordered = make_doc("nums: 1 2 3 4 5", metadata_text="nums: 5 4 3 2 1")
        self.assertTrue(counter_basic.filter(doc_reordered))
        self.assertTrue(counter_extended.filter(doc_reordered))

    def test_length_ratio_filter(self):
        length_filter_parallel = LengthRatioFilter(
            list_path=["text", "metadata['test_text']"],
            ratio_threshold=3.0,
        )
        self.assertTrue(
            length_filter_parallel.filter(
                make_doc(TEXT_EN_PARALLEL, metadata_text=TEXT_RU_PARALLEL)
            )
        )
        self.assertTrue(
            length_filter_parallel.filter(
                make_doc(TEXT_RU_PARALLEL, metadata_text=TEXT_FR_PARALLEL)
            )
        )

        length_filter = LengthRatioFilter(
            list_path=["text", "metadata['test_text']"],
            ratio_threshold=1.8,
        )

        doc_equal_len = make_doc("x" * 100)
        self.assertTrue(length_filter.filter(doc_equal_len))

        doc_equal_threshold = make_doc("a" * 100, metadata_text="b" * 180)
        self.assertTrue(length_filter.filter(doc_equal_threshold))

        doc_within_threshold = make_doc("a" * 100, metadata_text="b" * 179)
        self.assertTrue(length_filter.filter(doc_within_threshold))

        doc_over_threshold = make_doc("a" * 100, metadata_text="b" * 181)
        self.assertFalse(length_filter.filter(doc_over_threshold))

        doc_empty_side = make_doc("nonempty", metadata_text="")
        self.check_filter(length_filter, doc_empty_side, "empty_text")

        length_filter_words = LengthRatioFilter(
            list_path=["text", "metadata['test_text']"],
            ratio_threshold=2.0,
            calc_method="words",
        )

        doc_same_words = make_doc(
            "The quick brown fox jumps over the lazy dog",
            metadata_text="Быстрая коричневая лиса прыгает через ленивую собаку",
        )
        self.assertTrue(length_filter_words.filter(doc_same_words))

        doc_words_within = make_doc(
            "I went to the store yesterday and bought some groceries",
            metadata_text="Вчера я ходил в магазин и купил продукты",
        )
        self.assertTrue(length_filter_words.filter(doc_words_within))

        doc_words_over = make_doc(
            "Hello world", metadata_text="Привет, мир, как дела, что нового у тебя"
        )
        self.assertFalse(length_filter_words.filter(doc_words_over))

    @require_fasttext
    @require_hf_hub
    def test_language_filter(self):
        model_path = fasttext_model_get_path(
            hf_repo_name="cis-lmu/glotlid", filename="model.bin"
        )
        lang_filter = LanguageFilter(
            model_download_url=model_path,
            model_subfolder="fasttext_model",
            languages=["eng_Latn", "rus_Cyrl"],
            language_threshold=0.5,
        )

        doc_en = make_doc(TEXT_EN_PARALLEL)
        doc_ru = make_doc(TEXT_RU_PARALLEL)
        doc_fr = make_doc(TEXT_FR_PARALLEL)

        self.assertTrue(lang_filter.filter(doc_en))
        self.assertEqual(doc_en.metadata["language"], "eng_Latn")

        self.assertTrue(lang_filter.filter(doc_ru))
        self.assertEqual(doc_ru.metadata["language"], "rus_Cyrl")

        self.assertFalse(lang_filter.filter(doc_fr))
        self.assertEqual(doc_fr.metadata["language"], "fra_Latn")

    @pytest.mark.skipif(
        os.getenv("RUN_INTEGRATION_TESTS") != "1",
        reason="Integration tests are disabled",
    )
    @require_openai
    def test_bad_translations_filter(self):
        from openai import OpenAI

        # don't forget to start the Infinity server (inference/infinity).
        client = OpenAI(base_url="http://localhost:8888", api_key="dummy_key")

        bt_filter = BadTranslationsFilter(
            client=client,
            model_name="",  # the server ignores the model name
            list_path=["text", "metadata['test_text']"],
            sim_score=0.8,
            batch_size=6,
        )

        docs = [
            make_doc(TEXT_EN_PARALLEL, metadata_text=TEXT_RU_PARALLEL),
            make_doc(
                TEXT_EN_PARALLEL,
                metadata_text="Сегодня солнечная погода, и я люблю мороженое.",
            ),
            make_doc(
                "The cat is sleeping on the sofa.", metadata_text="Кот спит на диване."
            ),
            make_doc(
                "We installed three servers and upgraded the network firmware.",
                metadata_text="Мы купили три яблока и один апельсин.",
            ),
            make_doc("Hello world!", metadata_text="Привет, мир!"),
            make_doc(
                "He loves to play basketball on weekends.",
                metadata_text="Это статья о квантовой механике и матрицах плотности.",
            ),
        ]
        expected = [True, False, True, False, True, False]
        result = bt_filter.filter_batch(docs)

        self.assertTrue(all(isinstance(x, bool) for x in result))
        self.assertEqual(result, expected, f"Expected {expected}, got {result}")
