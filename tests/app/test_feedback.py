"""
Tests for jrm_advisor.app.feedback — SQL sanitiser, WS client routing,
UC identifier validation, and rating whitelist.

No live Databricks connection required.
"""

from __future__ import annotations

import pytest
from unittest.mock import MagicMock, patch

from jrm_advisor.app.feedback import _escape, _get_ws_client, _validate_uc_ident


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


class TestGetWsClient:
    def test_uses_user_token_when_provided(self):
        """_get_ws_client should use WorkspaceClient(host=..., token=...) for OBO."""
        with (
            patch("databricks.sdk.WorkspaceClient") as mock_ws_cls,
            patch.dict(
                "os.environ",
                {"DATABRICKS_HOST": "https://adb-test.azuredatabricks.net"},
            ),
        ):
            mock_ws_cls.return_value = MagicMock()
            _get_ws_client(user_token="dapi_obo_token")
            mock_ws_cls.assert_called_once_with(
                host="https://adb-test.azuredatabricks.net",
                token="dapi_obo_token",
            )

    def test_uses_config_when_no_token_local_dev(self):
        """_get_ws_client falls back to Config() in local dev (DATABRICKS_TOKEN set)."""
        with (
            patch("databricks.sdk.WorkspaceClient") as mock_ws_cls,
            patch("databricks.sdk.core.Config") as mock_config_cls,
            patch.dict("os.environ", {"DATABRICKS_TOKEN": "dapi_local_dev"}),
        ):
            mock_ws_cls.return_value = MagicMock()
            mock_config_cls.return_value = MagicMock()
            _get_ws_client(user_token=None)
            mock_config_cls.assert_called_once()
            mock_ws_cls.assert_called_once_with(config=mock_config_cls.return_value)

    def test_raises_when_no_token_and_not_local_dev(self):
        """_get_ws_client raises ValueError in production when no user_token is present."""
        env_without_token = {
            k: v
            for k, v in __import__("os").environ.items()
            if k not in {"DATABRICKS_TOKEN", "USE_MOCK_BACKEND"}
        }
        with patch.dict("os.environ", env_without_token, clear=True):
            with pytest.raises(ValueError, match="no user_token provided"):
                _get_ws_client(user_token=None)


class TestValidateUcIdent:
    """SEC-1: Unity Catalog identifier validation."""

    def test_valid_identifier_returned_unchanged(self):
        assert _validate_uc_ident("dsa_development", "catalog") == "dsa_development"

    def test_valid_identifier_with_numbers(self):
        assert (
            _validate_uc_ident("jrm_advisor_feedback2", "table")
            == "jrm_advisor_feedback2"
        )

    def test_hyphen_raises(self):
        with pytest.raises(ValueError, match="only alphanumeric"):
            _validate_uc_ident("my-catalog", "catalog")

    def test_dot_raises(self):
        with pytest.raises(ValueError, match="only alphanumeric"):
            _validate_uc_ident("cat.schema", "catalog")

    def test_semicolon_raises(self):
        with pytest.raises(ValueError, match="only alphanumeric"):
            _validate_uc_ident("table;DROP TABLE foo", "table")

    def test_space_raises(self):
        with pytest.raises(ValueError, match="only alphanumeric"):
            _validate_uc_ident("my catalog", "catalog")

    def test_empty_string_raises(self):
        # An empty string should not match the regex (requires at least one char)
        with pytest.raises(ValueError, match="only alphanumeric"):
            _validate_uc_ident("", "catalog")

    def test_label_included_in_error(self):
        with pytest.raises(ValueError, match="catalog"):
            _validate_uc_ident("bad-name", "catalog")


class TestRatingWhitelist:
    """SEC-3: rating must be 'thumbs_up' or 'thumbs_down'."""

    def test_invalid_rating_returns_false(self):
        from jrm_advisor.app.feedback import submit_feedback

        result = submit_feedback(
            question="q",
            answer_text="a",
            rating="DELETE FROM feedback",
        )
        assert result is False

    def test_empty_rating_returns_false(self):
        from jrm_advisor.app.feedback import submit_feedback

        result = submit_feedback(question="q", answer_text="a", rating="")
        assert result is False

    def test_thumbs_up_passes_whitelist(self, monkeypatch):
        """thumbs_up is valid — execution continues past the whitelist check."""
        from jrm_advisor.app.feedback import submit_feedback

        # No warehouse ID set → returns False after the whitelist check passes.
        monkeypatch.delenv("DATABRICKS_SQL_WAREHOUSE_ID", raising=False)
        result = submit_feedback(question="q", answer_text="a", rating="thumbs_up")
        # Should fail at warehouse_id check, not at rating whitelist
        assert result is False

    def test_thumbs_down_passes_whitelist(self, monkeypatch):
        from jrm_advisor.app.feedback import submit_feedback

        monkeypatch.delenv("DATABRICKS_SQL_WAREHOUSE_ID", raising=False)
        result = submit_feedback(question="q", answer_text="a", rating="thumbs_down")
        assert result is False
