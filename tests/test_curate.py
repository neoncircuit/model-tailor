"""Unit tests for the curate module (dedup, filter, balance, split).

All tests run without API keys, GPU, or external services.
Uses seed=42 throughout for deterministic, reproducible results.
"""

from __future__ import annotations

import pytest

from src.curate.balance import DatasetBalancer
from src.curate.dedup import ExactDedup, FuzzyDedup, _tokenize
from src.curate.filter import QualityFilter
from src.curate.split import DatasetSplitter

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def sample_records() -> list[dict[str, object]]:
    """Minimal NL/SQL dataset with a mix of duplicates and unique entries."""
    return [
        {"nl": "Show all users", "sql": "SELECT * FROM users", "score": 4.0, "difficulty": "easy"},
        {"nl": "Show all users", "sql": "SELECT * FROM users", "score": 4.0, "difficulty": "easy"},
        {
            "nl": "Count orders",
            "sql": "SELECT COUNT(*) FROM orders",
            "score": 5.0,
            "difficulty": "medium",
        },
        {
            "nl": "Get product names",
            "sql": "SELECT name FROM products",
            "score": 3.5,
            "difficulty": "easy",
        },
        {
            "nl": "Total revenue",
            "sql": "SELECT SUM(amount) FROM sales",
            "score": 2.0,
            "difficulty": "hard",
        },
        {
            "nl": "Count orders",
            "sql": "SELECT COUNT(*) FROM orders",
            "score": 5.0,
            "difficulty": "medium",
        },
    ]


@pytest.fixture()
def large_dataset() -> list[dict[str, object]]:
    """Larger dataset (100 records) for split and balance tests."""
    records = []
    difficulties = ["easy", "medium", "hard"]
    for i in range(100):
        diff = difficulties[i % 3]
        records.append(
            {
                "nl": f"Query number {i}",
                "sql": f"SELECT col{i} FROM table{i}",
                "score": 3.0 + (i % 5) * 0.5,
                "difficulty": diff,
            }
        )
    return records


# ===========================================================================
# _tokenize
# ===========================================================================


class TestTokenize:
    """Tests for the _tokenize helper function."""

    def test_lowercases_and_splits(self) -> None:
        """Lowercases input and splits on non-alphanumeric characters."""
        assert _tokenize("Hello World!") == ["hello", "world"]

    def test_splits_on_non_alnum(self) -> None:
        """Splits on punctuation, hyphens, and special characters."""
        result = _tokenize("foo-bar_baz.qux")
        assert result == ["foo", "bar_baz", "qux"]

    def test_empty_string(self) -> None:
        """Empty input returns empty list."""
        assert _tokenize("") == []

    def test_mixed_case_digits(self) -> None:
        """Digits and mixed case are handled correctly."""
        assert _tokenize("SELECT col1 FROM t2") == ["select", "col1", "from", "t2"]


# ===========================================================================
# ExactDedup
# ===========================================================================


class TestExactDedup:
    """Tests for ExactDedup."""

    def test_removes_exact_duplicates_keeps_first(self, sample_records: list[dict]) -> None:
        """Removes exact duplicates and keeps the first occurrence by default."""
        dedup = ExactDedup()
        result = dedup.run(sample_records)
        # Original has 6 records with 2 pairs of duplicates -> 4 unique
        assert len(result) == 4
        # First occurrence is preserved
        assert result[0] == sample_records[0]
        assert result[1] == sample_records[2]

    def test_keep_last(self, sample_records: list[dict]) -> None:
        """keep='last' retains the last occurrence of each duplicate."""
        dedup = ExactDedup(keep="last")
        result = dedup.run(sample_records)
        assert len(result) == 4
        # The last "Show all users" is at index 1, and last "Count orders" at index 5.
        # With keep="last", the output preserves original order of kept records.
        nl_values = [r["nl"] for r in result]
        assert "Show all users" in nl_values
        assert "Count orders" in nl_values

    def test_no_duplicates_returns_all(self) -> None:
        """When there are no duplicates, all records are returned."""
        data = [
            {"nl": "A", "sql": "SELECT 1"},
            {"nl": "B", "sql": "SELECT 2"},
            {"nl": "C", "sql": "SELECT 3"},
        ]
        dedup = ExactDedup()
        result = dedup.run(data)
        assert len(result) == 3
        assert result == data

    def test_single_field_dedup(self) -> None:
        """Dedup on a single field (sql) treats records with same SQL as duplicates."""
        data = [
            {"nl": "Query A", "sql": "SELECT * FROM users"},
            {"nl": "Query B", "sql": "SELECT * FROM users"},
            {"nl": "Query C", "sql": "SELECT * FROM orders"},
        ]
        dedup = ExactDedup(fields=["sql"])
        result = dedup.run(data)
        assert len(result) == 2
        assert result[0]["nl"] == "Query A"
        assert result[1]["nl"] == "Query C"


# ===========================================================================
# FuzzyDedup
# ===========================================================================


class TestFuzzyDedup:
    """Tests for FuzzyDedup."""

    def test_similar_texts_deduplicated(self) -> None:
        """Very similar NL texts are identified as near-duplicates."""
        data = [
            {"nl": "Count all users in the table", "sql": "SELECT COUNT(*) FROM users"},
            {"nl": "Count all the users in table", "sql": "SELECT COUNT(*) FROM users"},
            {
                "nl": "Get total revenue by region",
                "sql": "SELECT region, SUM(revenue) FROM sales GROUP BY region",
            },
        ]
        dedup = FuzzyDedup(field="nl", threshold=0.5, num_perm=128)
        result = dedup.run(data)
        # The two similar texts should collapse to one
        assert len(result) <= 2
        # The distinct query about revenue should always survive
        assert any("revenue" in r["nl"] for r in result)

    def test_different_texts_kept(self) -> None:
        """Completely different texts are all kept."""
        data = [
            {"nl": "Show all users", "sql": "SELECT * FROM users"},
            {
                "nl": "Calculate total revenue for Q3",
                "sql": "SELECT SUM(revenue) FROM sales WHERE quarter=3",
            },
            {
                "nl": "Find the oldest employee",
                "sql": "SELECT * FROM employees ORDER BY birth_date LIMIT 1",
            },
        ]
        dedup = FuzzyDedup(field="nl", threshold=0.85)
        result = dedup.run(data)
        assert len(result) == 3

    def test_invalid_threshold_raises(self) -> None:
        """Threshold outside (0, 1] raises ValueError."""
        with pytest.raises(ValueError, match="threshold"):
            FuzzyDedup(threshold=0.0)
        with pytest.raises(ValueError, match="threshold"):
            FuzzyDedup(threshold=1.5)
        with pytest.raises(ValueError, match="threshold"):
            FuzzyDedup(threshold=-0.1)


# ===========================================================================
# QualityFilter
# ===========================================================================


class TestQualityFilter:
    """Tests for QualityFilter."""

    def test_filter_length(self) -> None:
        """filter_length removes records outside the [min, max] character range."""
        data = [
            {"nl": "Hi", "sql": "SELECT 1"},  # nl too short (2 chars)
            {"nl": "Show all users", "sql": "SELECT * FROM users"},  # both OK
            {"nl": "A" * 600, "sql": "SELECT 1"},  # nl too long
        ]
        result = QualityFilter(data).filter_length(min_len=5, max_len=500).result()
        assert len(result) == 1
        assert result[0]["nl"] == "Show all users"

    def test_filter_sql_valid(self) -> None:
        """filter_sql_valid removes records with invalid SQL."""
        data = [
            {"nl": "Good query", "sql": "SELECT * FROM users"},
            {"nl": "Empty sql", "sql": ""},
            {"nl": "Whitespace sql", "sql": "   "},
            {"nl": "Valid insert", "sql": "INSERT INTO t VALUES (1)"},
        ]
        result = QualityFilter(data).filter_sql_valid().result()
        # Only the valid SQL records survive
        assert len(result) == 2
        assert result[0]["sql"] == "SELECT * FROM users"
        assert result[1]["sql"] == "INSERT INTO t VALUES (1)"

    def test_filter_has_keywords(self) -> None:
        """filter_has_keywords removes records missing required SQL keywords."""
        data = [
            {"nl": "A", "sql": "SELECT name FROM users"},
            {"nl": "B", "sql": "INSERT INTO users VALUES (1)"},  # no SELECT/FROM pair
            {"nl": "C", "sql": "SELECT * FROM orders WHERE id=1"},
        ]
        result = QualityFilter(data).filter_has_keywords().result()
        assert len(result) == 2
        assert all("SELECT" in r["sql"] and "FROM" in r["sql"] for r in result)

    def test_filter_score(self) -> None:
        """filter_score removes records below the minimum score threshold."""
        data = [
            {"nl": "A", "sql": "S1", "score": 4.5},
            {"nl": "B", "sql": "S2", "score": 2.0},
            {"nl": "C", "sql": "S3", "score": 3.0},
            {"nl": "D", "sql": "S4"},  # no score -> kept (lenient)
        ]
        result = QualityFilter(data).filter_score(min_score=3.0).result()
        assert len(result) == 3
        # "B" with score 2.0 is removed
        assert all(r["nl"] != "B" for r in result)

    def test_chaining_filters(self) -> None:
        """Multiple filters can be chained and all apply cumulatively."""
        data = [
            {"nl": "Show all users", "sql": "SELECT * FROM users", "score": 4.0},
            {"nl": "Hi", "sql": "SELECT 1 FROM t", "score": 5.0},  # nl too short
            {"nl": "Get orders", "sql": "NOT VALID SQL ???", "score": 4.0},  # bad SQL keywords
            {"nl": "Revenue report", "sql": "SELECT SUM(x) FROM sales", "score": 1.0},  # low score
        ]
        result = (
            QualityFilter(data)
            .filter_length(min_len=5, max_len=500)
            .filter_has_keywords()
            .filter_score(min_score=3.0)
            .result()
        )
        assert len(result) == 1
        assert result[0]["nl"] == "Show all users"

    def test_result_returns_filtered_list(self) -> None:
        """result() returns the surviving records as a plain list."""
        data = [
            {"nl": "Show users", "sql": "SELECT * FROM users"},
        ]
        result = QualityFilter(data).result()
        assert isinstance(result, list)
        assert len(result) == 1


# ===========================================================================
# DatasetBalancer
# ===========================================================================


class TestDatasetBalancer:
    """Tests for DatasetBalancer."""

    def test_balances_to_target_distribution(self) -> None:
        """Balancing adjusts category counts toward the target proportions."""
        data = [
            {"nl": f"easy-{i}", "sql": f"SELECT {i}", "difficulty": "easy"} for i in range(50)
        ] + [{"nl": f"hard-{i}", "sql": f"SELECT {i}", "difficulty": "hard"} for i in range(10)]
        balancer = DatasetBalancer(
            field="difficulty",
            target={"easy": 0.5, "hard": 0.5},
            strategy="both",
            seed=42,
        )
        result = balancer.run(data)
        # Count each category in the result
        easy_count = sum(1 for r in result if r["difficulty"] == "easy")
        hard_count = sum(1 for r in result if r["difficulty"] == "hard")
        # With "both" strategy, proportions should be close to 50/50
        total = easy_count + hard_count
        assert total > 0
        easy_ratio = easy_count / total
        assert 0.35 <= easy_ratio <= 0.65, f"Easy ratio {easy_ratio} not near 0.5"

    def test_report_returns_correct_info(self) -> None:
        """report() returns field, total, counts, distribution, and target."""
        data = [
            {"difficulty": "easy"},
            {"difficulty": "easy"},
            {"difficulty": "hard"},
        ]
        balancer = DatasetBalancer(
            field="difficulty",
            target={"easy": 0.7, "hard": 0.3},
            seed=42,
        )
        report = balancer.report(data)
        assert report["field"] == "difficulty"
        assert report["total"] == 3
        assert report["counts"]["easy"] == 2
        assert report["counts"]["hard"] == 1
        assert "distribution" in report
        assert "target" in report
        assert abs(report["distribution"]["easy"] - 2 / 3) < 0.01

    def test_none_target_raises(self) -> None:
        """Passing target=None raises ValueError."""
        with pytest.raises(ValueError, match="target"):
            DatasetBalancer(field="difficulty", target=None)

    def test_empty_target_raises(self) -> None:
        """Passing target with zero-sum proportions raises ValueError."""
        with pytest.raises(ValueError, match="positive"):
            DatasetBalancer(field="difficulty", target={"easy": 0, "hard": 0})


# ===========================================================================
# DatasetSplitter
# ===========================================================================


class TestDatasetSplitter:
    """Tests for DatasetSplitter."""

    def test_default_split_sizes(self, large_dataset: list[dict]) -> None:
        """Default 80/10/10 split produces correct approximate sizes."""
        splitter = DatasetSplitter(seed=42)
        splits = splitter.run(large_dataset)
        total = sum(len(v) for v in splits.values())
        assert total == 100
        assert len(splits["train"]) == 80
        assert len(splits["val"]) == 10
        assert len(splits["test"]) == 10

    def test_stratified_split_preserves_distribution(self, large_dataset: list[dict]) -> None:
        """Stratified split preserves category distribution across splits."""
        splitter = DatasetSplitter(stratify_field="difficulty", seed=42)
        splits = splitter.run(large_dataset)
        total = sum(len(v) for v in splits.values())
        assert total == 100

        # Each split should contain all three difficulty levels
        for name in ("train", "val", "test"):
            difficulties = {r["difficulty"] for r in splits[name]}
            assert difficulties == {"easy", "medium", "hard"}, (
                f"Split '{name}' missing difficulties: {difficulties}"
            )

    def test_deterministic_with_same_seed(self, large_dataset: list[dict]) -> None:
        """Same seed produces identical splits."""
        splitter1 = DatasetSplitter(seed=42)
        splitter2 = DatasetSplitter(seed=42)
        splits1 = splitter1.run(large_dataset)
        splits2 = splitter2.run(large_dataset)
        for name in ("train", "val", "test"):
            assert splits1[name] == splits2[name]

    def test_stats_returns_correct_counts(self, large_dataset: list[dict]) -> None:
        """stats() returns total and per-split count and ratio information."""
        splitter = DatasetSplitter(seed=42)
        splits = splitter.run(large_dataset)
        st = DatasetSplitter.stats(splits)
        assert st["total"] == 100
        assert st["splits"]["train"]["count"] == 80
        assert st["splits"]["val"]["count"] == 10
        assert st["splits"]["test"]["count"] == 10
        assert abs(st["splits"]["train"]["ratio"] - 0.8) < 0.01

    def test_stats_with_field(self, large_dataset: list[dict]) -> None:
        """stats() with a field returns per-split category distributions."""
        splitter = DatasetSplitter(stratify_field="difficulty", seed=42)
        splits = splitter.run(large_dataset)
        st = DatasetSplitter.stats(splits, field="difficulty")
        # Train split should have distribution info for each difficulty
        train_dist = st["splits"]["train"]["distribution"]
        assert "easy" in train_dist
        assert "medium" in train_dist
        assert "hard" in train_dist

    def test_invalid_ratios_raises(self) -> None:
        """Ratios that sum to zero or negative raise ValueError."""
        with pytest.raises(ValueError, match="positive"):
            DatasetSplitter(ratios={"train": 0, "val": 0, "test": 0})
        with pytest.raises(ValueError, match="positive"):
            DatasetSplitter(ratios={"train": -1, "val": 0, "test": 0})
