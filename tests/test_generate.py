"""Unit tests for the generate module (few_shot and quality).

All tests run without API keys, GPU, or external services.
QualityChecker is instantiated with client=None to skip LLM calls.
"""

from __future__ import annotations

import json

import pytest

from src.generate.few_shot import FewShotFormatter, SeedExample, SeedLoader
from src.generate.quality import QualityChecker, QualityResult

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def sample_seeds() -> list[SeedExample]:
    """Return a small list of SeedExample objects for testing."""
    return [
        SeedExample(
            natural_language="Show all employees in the sales department",
            sql="SELECT * FROM employees WHERE department = 'sales';",
            difficulty="easy",
            category="select",
        ),
        SeedExample(
            natural_language="Count orders grouped by customer with more than five orders",
            sql="SELECT customer_id, COUNT(*) FROM orders "
            "GROUP BY customer_id HAVING COUNT(*) > 5;",
            difficulty="medium",
            category="aggregation",
        ),
        SeedExample(
            natural_language="Find the top 3 products by revenue using a window function",
            sql=(
                "SELECT product_id, revenue, "
                "RANK() OVER (ORDER BY revenue DESC) AS rnk "
                "FROM products LIMIT 3;"
            ),
            difficulty="hard",
            category="window",
        ),
    ]


@pytest.fixture()
def seed_jsonl(tmp_path) -> str:
    """Write a temporary JSONL seed file and return its path."""
    records = [
        {
            "natural_language": "List all users",
            "sql": "SELECT * FROM users;",
            "difficulty": "easy",
            "category": "select",
        },
        {
            "natural_language": "Get total revenue per month",
            "sql": "SELECT month, SUM(revenue) FROM sales GROUP BY month;",
            "difficulty": "medium",
            "category": "aggregation",
        },
        {
            "natural_language": "Find duplicate emails in the contacts table",
            "sql": ("SELECT email, COUNT(*) FROM contacts GROUP BY email HAVING COUNT(*) > 1;"),
            "difficulty": "hard",
            "category": "aggregation",
        },
    ]
    path = tmp_path / "seeds.jsonl"
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")
    return str(path)


@pytest.fixture()
def checker_no_llm() -> QualityChecker:
    """Return a QualityChecker with no LLM client (heuristic + syntax only)."""
    return QualityChecker(client=None)


@pytest.fixture()
def valid_pair() -> dict:
    """Return a valid NL-SQL pair that passes all heuristic and syntax checks."""
    return {
        "natural_language": "Show all employees in the engineering department",
        "sql": "SELECT * FROM employees WHERE department = 'engineering';",
    }


# ===========================================================================
# SeedExample tests
# ===========================================================================


class TestSeedExample:
    """Tests for the SeedExample dataclass."""

    def test_to_dict_returns_correct_keys(self) -> None:
        """to_dict() must return a dict with the four documented keys."""
        ex = SeedExample(
            natural_language="What is the average salary?",
            sql="SELECT AVG(salary) FROM employees;",
            difficulty="medium",
            category="aggregation",
        )
        d = ex.to_dict()
        assert d == {
            "natural_language": "What is the average salary?",
            "sql": "SELECT AVG(salary) FROM employees;",
            "difficulty": "medium",
            "category": "aggregation",
        }

    def test_to_dict_default_values(self) -> None:
        """to_dict() uses defaults for optional fields."""
        ex = SeedExample(natural_language="q", sql="SELECT 1;")
        d = ex.to_dict()
        assert d["difficulty"] == "medium"
        assert d["category"] == ""


# ===========================================================================
# SeedLoader tests
# ===========================================================================


class TestSeedLoader:
    """Tests for the SeedLoader class."""

    def test_load_parses_jsonl(self, seed_jsonl: str) -> None:
        """load() should parse every valid line into a SeedExample."""
        loader = SeedLoader(seed_jsonl)
        examples = loader.load()
        assert len(examples) == 3
        assert examples[0].natural_language == "List all users"
        assert examples[2].difficulty == "hard"

    def test_load_raises_on_missing_file(self, tmp_path) -> None:
        """load() must raise FileNotFoundError for a non-existent path."""
        loader = SeedLoader(tmp_path / "nonexistent.jsonl")
        with pytest.raises(FileNotFoundError):
            loader.load()

    def test_filter_by_difficulty(self, seed_jsonl: str) -> None:
        """filter_by_difficulty() returns only matching examples."""
        loader = SeedLoader(seed_jsonl)
        loader.load()
        easy = loader.filter_by_difficulty("easy")
        assert len(easy) == 1
        assert easy[0].difficulty == "easy"

    def test_filter_by_category(self, seed_jsonl: str) -> None:
        """filter_by_category() returns only matching examples."""
        loader = SeedLoader(seed_jsonl)
        loader.load()
        agg = loader.filter_by_category("aggregation")
        assert len(agg) == 2
        assert all(ex.category == "aggregation" for ex in agg)

    def test_to_dicts_returns_list_of_dicts(self, seed_jsonl: str) -> None:
        """to_dicts() returns a list of plain dicts."""
        loader = SeedLoader(seed_jsonl)
        loader.load()
        dicts = loader.to_dicts()
        assert isinstance(dicts, list)
        assert len(dicts) == 3
        for d in dicts:
            assert isinstance(d, dict)
            assert "natural_language" in d
            assert "sql" in d


# ===========================================================================
# FewShotFormatter tests
# ===========================================================================


class TestFewShotFormatter:
    """Tests for the FewShotFormatter class."""

    def test_format_numbered_style(self, sample_seeds: list[SeedExample]) -> None:
        """format() with style='numbered' produces expected structure."""
        formatter = FewShotFormatter(sample_seeds)
        output = formatter.format(n=2, style="numbered")
        assert "Example 1:" in output
        assert "Example 2:" in output
        assert "Question:" in output
        assert "SQL:" in output

    def test_format_json_style_is_valid_json(self, sample_seeds: list[SeedExample]) -> None:
        """format() with style='json' returns parseable JSON."""
        formatter = FewShotFormatter(sample_seeds)
        output = formatter.format(n=2, style="json")
        data = json.loads(output)
        assert isinstance(data, list)
        assert len(data) == 2
        assert "natural_language" in data[0]

    def test_format_empty_examples_returns_fallback(self) -> None:
        """format() with no examples returns the fallback message."""
        formatter = FewShotFormatter([])
        output = formatter.format(n=3)
        assert output == "(No few-shot examples available.)"

    def test_select_respects_n_parameter(self, sample_seeds: list[SeedExample]) -> None:
        """select() returns at most n examples."""
        formatter = FewShotFormatter(sample_seeds)
        selected = formatter.select(n=1)
        assert len(selected) == 1
        selected_all = formatter.select(n=10)
        assert len(selected_all) == len(sample_seeds)


# ===========================================================================
# QualityChecker — heuristic checks
# ===========================================================================


class TestHeuristicChecks:
    """Tests for QualityChecker._heuristic_checks."""

    def test_valid_pair_passes(self, checker_no_llm: QualityChecker) -> None:
        """A well-formed NL-SQL pair should pass all heuristics."""
        issues: list[str] = []
        result = checker_no_llm._heuristic_checks(
            nl="Show all employees in the engineering department",
            sql="SELECT * FROM employees WHERE department = 'engineering';",
            issues=issues,
        )
        assert result is True
        assert issues == []

    def test_nl_too_short_fails(self, checker_no_llm: QualityChecker) -> None:
        """NL shorter than min_nl_length should fail."""
        issues: list[str] = []
        result = checker_no_llm._heuristic_checks(
            nl="Hi",
            sql="SELECT * FROM employees;",
            issues=issues,
        )
        assert result is False
        assert any("NL too short" in i for i in issues)

    def test_sql_missing_keywords_fails(self, checker_no_llm: QualityChecker) -> None:
        """SQL without any recognised keyword should fail."""
        issues: list[str] = []
        result = checker_no_llm._heuristic_checks(
            nl="This is a perfectly valid natural language question",
            sql="some random text that is not sql at all here",
            issues=issues,
        )
        assert result is False
        assert any("missing standard keywords" in i for i in issues)

    def test_nl_looks_like_sql_fails(self, checker_no_llm: QualityChecker) -> None:
        """NL that starts with a SQL keyword should fail."""
        issues: list[str] = []
        result = checker_no_llm._heuristic_checks(
            nl="SELECT * FROM employees WHERE department = 'engineering'",
            sql="SELECT * FROM employees WHERE department = 'engineering';",
            issues=issues,
        )
        assert result is False
        assert any("looks like raw SQL" in i for i in issues)

    def test_unbalanced_parentheses_fails(self, checker_no_llm: QualityChecker) -> None:
        """SQL with unbalanced parentheses should fail."""
        issues: list[str] = []
        result = checker_no_llm._heuristic_checks(
            nl="Count the number of active users in each region",
            sql="SELECT region, COUNT(*  FROM users WHERE active = 1 GROUP BY region;",
            issues=issues,
        )
        assert result is False
        assert any("unbalanced parentheses" in i for i in issues)


# ===========================================================================
# QualityChecker — syntax check
# ===========================================================================


class TestSyntaxCheck:
    """Tests for QualityChecker._syntax_check."""

    def test_valid_sql_passes(self, checker_no_llm: QualityChecker) -> None:
        """Well-formed SQL should pass the syntax check."""
        issues: list[str] = []
        result = checker_no_llm._syntax_check(
            "SELECT id, name FROM users WHERE active = 1;",
            issues,
        )
        assert result is True
        assert issues == []

    def test_invalid_sql_fails(self, checker_no_llm: QualityChecker) -> None:
        """SQL ending with a trailing comma should fail the syntax check."""
        issues: list[str] = []
        result = checker_no_llm._syntax_check(
            "SELECT id, name,",
            issues,
        )
        assert result is False
        assert any("trailing comma" in i for i in issues)


# ===========================================================================
# QualityChecker — _parse_score
# ===========================================================================


class TestParseScore:
    """Tests for QualityChecker._parse_score."""

    def test_json_response_extracts_score(self, checker_no_llm: QualityChecker) -> None:
        """A JSON response with a 'score' key is parsed correctly."""
        response = '{"score": 4, "reason": "Good quality pair"}'
        assert checker_no_llm._parse_score(response) == 4.0

    def test_bare_number_extracts_score(self, checker_no_llm: QualityChecker) -> None:
        """A plain-text response containing a digit 1-5 is extracted."""
        response = "I would rate this a 5 out of 5."
        assert checker_no_llm._parse_score(response) == 5.0

    def test_clamping_high(self, checker_no_llm: QualityChecker) -> None:
        """Scores above 5 are clamped to 5.0."""
        response = '{"score": 10}'
        assert checker_no_llm._parse_score(response) == 5.0

    def test_clamping_low(self, checker_no_llm: QualityChecker) -> None:
        """Scores below 1 are clamped to 1.0."""
        response = '{"score": -2}'
        assert checker_no_llm._parse_score(response) == 1.0


# ===========================================================================
# QualityResult
# ===========================================================================


class TestQualityResult:
    """Tests for QualityResult.summary()."""

    def test_summary_includes_pass(self) -> None:
        """summary() for a passing result includes '[PASS]'."""
        result = QualityResult(passed=True)
        assert "[PASS]" in result.summary()

    def test_summary_includes_fail(self) -> None:
        """summary() for a failing result includes '[FAIL]'."""
        result = QualityResult(passed=False, heuristic_ok=False, issues=["too short"])
        s = result.summary()
        assert "[FAIL]" in s
        assert "heuristic_fail" in s
        assert "too short" in s


# ===========================================================================
# QualityChecker — integration-level (no LLM)
# ===========================================================================


class TestQualityCheckerNoLLM:
    """Integration tests using QualityChecker with client=None."""

    def test_check_valid_pair_passes(
        self, checker_no_llm: QualityChecker, valid_pair: dict
    ) -> None:
        """check() on a valid pair returns a passing QualityResult."""
        result = checker_no_llm.check(valid_pair)
        assert result.passed is True
        assert result.heuristic_ok is True
        assert result.syntax_ok is True
        assert result.llm_score is None

    def test_filter_passed_returns_only_passing(
        self, checker_no_llm: QualityChecker, valid_pair: dict
    ) -> None:
        """filter_passed() keeps good examples and drops bad ones."""
        bad_pair = {"natural_language": "Hi", "sql": "nope"}
        examples = [valid_pair, bad_pair]
        passed = checker_no_llm.filter_passed(examples)
        assert len(passed) == 1
        assert passed[0] is valid_pair
