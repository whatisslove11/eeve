import unittest

from eeve.data.formatters import (
    CallableFormatter,
    FTFYFormatter,
)

from .utils import require_ftfy


def collapse_whitespace(s: str) -> str:
    return " ".join(s.split())


def reverse_words(s: str) -> str:
    return " ".join(w[::-1] for w in s.split())


def to_uppercase(s: str) -> str:
    return s.upper()


class TestFormatters(unittest.TestCase):
    @require_ftfy
    def test_ftfy_formatter(self):
        text_in = "Tom &amp; Jerry met at the caf\u00c3\u00a9 during El Ni\u00c3\u00b1o. The sign was \x1b[31mred\x1b[0m."
        ftfy_fmt = FTFYFormatter()
        text_out = ftfy_fmt.format(text_in)
        expected = "Tom & Jerry met at the café during El Niño. The sign was red."
        self.assertEqual(text_out, expected)

        text_html = "Fish &amp; Chips and 2 &lt; 3 &gt; 1"
        ftfy_no_html = FTFYFormatter(unescape_html=False)
        text_no_html_out = ftfy_no_html.format(text_html)
        self.assertEqual(text_no_html_out, text_html)

        text_ansi = "Status: \x1b[32mOK\x1b[0m and \x1b[31mFAIL\x1b[0m"
        ftfy_keep_ansi = FTFYFormatter(
            remove_terminal_escapes=False, remove_control_chars=False
        )
        text_keep_ansi_out = ftfy_keep_ansi.format(text_ansi)
        self.assertEqual(text_keep_ansi_out, text_ansi)

        text_mojibake = "caf\u00c3\u00a9 & El Ni\u00c3\u00b1o"
        ftfy_basic = FTFYFormatter()
        text_mojibake_out = ftfy_basic.format(text_mojibake)
        self.assertEqual(text_mojibake_out, "café & El Niño")

    def test_callable_formatter(self):
        fmt_collapse = CallableFormatter(func=collapse_whitespace)
        fmt_reverse = CallableFormatter(func=reverse_words)
        fmt_upper = CallableFormatter(func=to_uppercase)

        raw_ws = "  Hello,   world  \n and \t\t\tsome\tspaces  "
        collapsed = fmt_collapse.format(raw_ws)
        self.assertEqual(collapsed, "Hello, world and some spaces")

        reversed_text = fmt_reverse.format("Hello world!")
        self.assertEqual(reversed_text, "olleH !dlrow")

        uppercased = fmt_upper.format("Café naïve coöperate São Tomé")
        self.assertEqual(uppercased, "CAFÉ NAÏVE COÖPERATE SÃO TOMÉ")
