"""Tests for src/evaluate/metrics.py.

All tests run without API keys, GPU, or external services.
"""

from __future__ import annotations

import pytest

from src.evaluate.metrics import (
    _normalize_sql,
    _normalize_text,
    bleu_score,
    evaluate_batch,
    exact_match,
    rouge_scores,
)

# ======================================================================
# Normalisation helpers
# ======================================================================


class TestNormalizeSql:
    """Tests for _normalize_sql()."""

    def test_removes_trailing_semicolon(self) -> None:
        """Trailing semicolons are stripped."""
        result = _normalize_sql("SELECT 1;")
        assert not result.endswith(";")

    def test_collapses_whitespace(self) -> None:
        """Multiple whitespace characters collapse to a single space."""
        result = _normalize_sql("SELECT   id   FROM   users")
        assert "   " not in result
        assert "select id from users" == result

    def test_lowercases(self) -> None:
        """SQL is lowercased."""
        result = _normalize_sql("SELECT Id FROM Users")
        assert result == result.lower()

    def test_strips_surrounding_whitespace(self) -> None:
        """Leading and trailing whitespace is stripped."""
        result = _normalize_sql("  SELECT 1  ")
        assert result == result.strip()


class TestNormalizeText:
    """Tests for _normalize_text()."""

    def test_lowercases(self) -> None:
        """Text is lowercased."""
        assert _normalize_text("Hello World") == "hello world"

    def test_collapses_whitespace(self) -> None:
        """Multiple whitespace characters collapse to a single space."""
        assert _normalize_text("hello   world") == "hello world"

    def test_strips(self) -> None:
        """Leading and trailing whitespace is stripped."""
        assert _normalize_text("  hello  ") == "hello"


# ======================================================================
# Individual metrics
# ======================================================================


class TestExactMatch:
    """Tests for exact_match()."""

    def test_identical_after_normalization(self) -> None:
        """Identical SQL (modulo casing/whitespace/semicolons) returns 1.0."""
        assert exact_match("SELECT * FROM users;", "select *  from  users") == 1.0

    def test_different_sql(self) -> None:
        """Different SQL returns 0.0."""
        assert exact_match("SELECT * FROM users", "SELECT * FROM orders") == 0.0


class TestBleuScore:
    """Tests for bleu_score()."""

    def test_identical_strings(self) -> None:
        """Identical strings return a BLEU score of 1.0."""
        score = bleu_score("SELECT id FROM users", "SELECT id FROM users")
        assert score == pytest.approx(1.0)

    def test_completely_different(self) -> None:
        """Completely different strings return a BLEU score near 0."""
        score = bleu_score(
            "alpha bravo charlie delta",
            "xylophone zebra quantum neutron",
        )
        assert score < 0.1


class TestRougeScores:
    """Tests for rouge_scores()."""

    def test_returns_expected_keys(self) -> None:
        """rouge_scores() returns dict with rouge1, rouge2, rougeL keys."""
        scores = rouge_scores("hello world", "hello world")
        assert "rouge1" in scores
        assert "rouge2" in scores
        assert "rougeL" in scores

    def test_identical_strings(self) -> None:
        """Identical strings return 1.0 for all ROUGE variants."""
        scores = rouge_scores("hello world", "hello world")
        assert scores["rouge1"] == pytest.approx(1.0)
        assert scores["rouge2"] == pytest.approx(1.0)
        assert scores["rougeL"] == pytest.approx(1.0)


# ======================================================================
# Batch evaluator
# ======================================================================


class TestEvaluateBatch:
    """Tests for evaluate_batch()."""

    def test_mean_scores(self) -> None:
        """Correct mean scores across multiple predictions."""
        preds = ["SELECT * FROM users", "SELECT id FROM orders"]
        targets = ["SELECT * FROM users", "SELECT id FROM orders"]

        result = evaluate_batch(preds, targets, metrics=["exact_match"])
        assert result["exact_match"] == pytest.approx(1.0)

    def test_mean_scores_mixed(self) -> None:
        """Mean is computed correctly when some match and some do not."""
        preds = ["SELECT * FROM users", "SELECT name FROM products"]
        targets = ["SELECT * FROM users", "SELECT id FROM orders"]

        result = evaluate_batch(preds, targets, metrics=["exact_match"])
        assert result["exact_match"] == pytest.approx(0.5)

    def test_custom_metrics_list(self) -> None:
        """Only requested metrics appear in the result."""
        preds = ["SELECT 1"]
        targets = ["SELECT 1"]

        result = evaluate_batch(preds, targets, metrics=["bleu"])
        assert "bleu" in result
        assert "exact_match" not in result

    def test_length_mismatch_raises(self) -> None:
        """Mismatched prediction/target lengths raise ValueError."""
        with pytest.raises(ValueError, match="same length"):
            evaluate_batch(["a", "b"], ["a"])

    def test_empty_lists(self) -> None:
        """Empty input lists return an empty dict."""
        result = evaluate_batch([], [])
        assert result == {}
