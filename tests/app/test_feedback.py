"""
Tests for jrm_advisor.app.feedback._escape — SQL sanitiser.

No Databricks connection required.
"""

from __future__ import annotations

from jrm_advisor.app.feedback import _escape


class TestEscape:
    def test_single_quote_escaped(self):
        assert _escape("O'Brien") == "O''Brien"

    def test_backslash_escaped(self):
        assert _escape("path\\to\\file") == "path\\\\to\\\\file"

    def test_both_escaped(self):
        assert _escape("it\\'s") == "it\\\\''s"

    def test_clean_string_unchanged(self):
        assert _escape("Heineken Pilsner W4 2024") == "Heineken Pilsner W4 2024"

    def test_empty_string(self):
        assert _escape("") == ""

    def test_control_characters_removed(self):
        # Null byte, carriage return, form feed
        result = _escape("hello\x00world\r\x0c")
        assert "\x00" not in result
        assert "\r" not in result
        assert "\x0c" not in result
        assert "helloworld" in result

    def test_unicode_format_chars_removed(self):
        # Zero-width non-joiner (U+200C) is category Cf
        result = _escape("hello\u200cworld")
        assert "\u200c" not in result
        assert "helloworld" in result

    def test_normal_unicode_preserved(self):
        # Accented characters and CJK are fine
        result = _escape("caf\u00e9 \u4e2d\u6587")
        assert "caf\u00e9" in result
        assert "\u4e2d\u6587" in result

    def test_multiple_quotes(self):
        assert _escape("it's a 'test'") == "it''s a ''test''"
