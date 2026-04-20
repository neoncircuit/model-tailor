"""Quality control for generated NL->SQL training examples.

Validates generated pairs using three layers:
1. Heuristic checks — fast, rule-based sanity filters.
2. SQL syntax validation — uses sqlparse to check well-formedness.
3. LLM-based scoring — asks the teacher model to rate quality 1-5.

Usage:
    checker = QualityChecker(client)
    result = checker.check(example)
    if result.passed:
        # keep the example
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field

import sqlparse

from src.llm.client import Message, TeacherClient

logger = logging.getLogger(__name__)

# SQL keywords that should appear in any valid SQL query
SQL_KEYWORDS = {
    "SELECT",
    "INSERT",
    "UPDATE",
    "DELETE",
    "CREATE",
    "ALTER",
    "DROP",
    "WITH",  # CTEs
}

# Minimum/maximum lengths (characters) for NL and SQL
DEFAULT_MIN_NL_LENGTH = 10
DEFAULT_MAX_NL_LENGTH = 500
DEFAULT_MIN_SQL_LENGTH = 10
DEFAULT_MAX_SQL_LENGTH = 2000

# Threshold for LLM quality score (1-5 scale)
DEFAULT_MIN_LLM_SCORE = 3


@dataclass
class QualityResult:
    """Result of quality checking a single example."""

    passed: bool
    heuristic_ok: bool = True
    syntax_ok: bool = True
    llm_score: float | None = None
    issues: list[str] = field(default_factory=list)

    def summary(self) -> str:
        """Build a human-readable one-line summary of this quality result.

        Returns:
            A bracketed status string including pass/fail flags, the LLM
            score (when available), and any recorded issues.
        """
        status = "PASS" if self.passed else "FAIL"
        parts = [f"[{status}]"]
        if not self.heuristic_ok:
            parts.append("heuristic_fail")
        if not self.syntax_ok:
            parts.append("syntax_fail")
        if self.llm_score is not None:
            parts.append(f"llm_score={self.llm_score:.1f}")
        if self.issues:
            parts.append(f"issues=[{', '.join(self.issues)}]")
        return " ".join(parts)


class QualityChecker:
    """Multi-layer quality checker for NL->SQL pairs.

    Args:
        client: TeacherClient for LLM-based scoring. If None, LLM scoring
            is skipped.
        min_nl_length: Minimum character length for natural language.
        max_nl_length: Maximum character length for natural language.
        min_sql_length: Minimum character length for SQL.
        max_sql_length: Maximum character length for SQL.
        min_llm_score: Minimum LLM quality score (1-5) to pass.
        require_valid_sql: Whether to fail examples with invalid SQL syntax.
        run_llm_check: Whether to run the LLM quality scoring step.
    """

    def __init__(
        self,
        client: TeacherClient | None = None,
        min_nl_length: int = DEFAULT_MIN_NL_LENGTH,
        max_nl_length: int = DEFAULT_MAX_NL_LENGTH,
        min_sql_length: int = DEFAULT_MIN_SQL_LENGTH,
        max_sql_length: int = DEFAULT_MAX_SQL_LENGTH,
        min_llm_score: float = DEFAULT_MIN_LLM_SCORE,
        require_valid_sql: bool = True,
        run_llm_check: bool = True,
    ) -> None:
        """Initialize the quality checker.

        Args:
            client: TeacherClient for LLM-based scoring. If ``None``, LLM
                scoring is skipped.
            min_nl_length: Minimum character length for natural language.
            max_nl_length: Maximum character length for natural language.
            min_sql_length: Minimum character length for SQL.
            max_sql_length: Maximum character length for SQL.
            min_llm_score: Minimum LLM quality score (1-5) to pass.
            require_valid_sql: Whether to fail examples with invalid SQL syntax.
            run_llm_check: Whether to run the LLM quality scoring step.
        """
        self.client = client
        self.min_nl_length = min_nl_length
        self.max_nl_length = max_nl_length
        self.min_sql_length = min_sql_length
        self.max_sql_length = max_sql_length
        self.min_llm_score = min_llm_score
        self.require_valid_sql = require_valid_sql
        self.run_llm_check = run_llm_check and (client is not None)

    def check(self, example: dict, schema: str | None = None) -> QualityResult:
        """Run all quality checks on a single NL->SQL example.

        Args:
            example: Dict with 'natural_language' and 'sql' keys.
            schema: Optional SQL schema for context in LLM scoring.

        Returns:
            QualityResult with pass/fail and details.
        """
        nl = example.get("natural_language", "")
        sql = example.get("sql", "")

        issues: list[str] = []

        # Layer 1: Heuristic checks
        heuristic_ok = self._heuristic_checks(nl, sql, issues)

        # Layer 2: SQL syntax validation
        syntax_ok = self._syntax_check(sql, issues)

        # Layer 3: LLM-based scoring (most expensive, run last)
        llm_score = None
        if self.run_llm_check and heuristic_ok and syntax_ok:
            llm_score = self._llm_score(nl, sql, schema)

        # Determine overall pass/fail
        passed = heuristic_ok
        if self.require_valid_sql:
            passed = passed and syntax_ok
        if llm_score is not None:
            passed = passed and (llm_score >= self.min_llm_score)

        return QualityResult(
            passed=passed,
            heuristic_ok=heuristic_ok,
            syntax_ok=syntax_ok,
            llm_score=llm_score,
            issues=issues,
        )

    def check_batch(self, examples: list[dict], schema: str | None = None) -> list[QualityResult]:
        """Check a batch of examples.

        Args:
            examples: List of dicts, each with 'natural_language' and 'sql' keys.
            schema: Optional SQL schema for context in LLM scoring.

        Returns:
            A list of QualityResult objects in the same order as the input.
        """
        return [self.check(ex, schema=schema) for ex in examples]

    def filter_passed(self, examples: list[dict], schema: str | None = None) -> list[dict]:
        """Return only examples that pass all quality checks.

        Args:
            examples: List of dicts, each with 'natural_language' and 'sql' keys.
            schema: Optional SQL schema for context in LLM scoring.

        Returns:
            A filtered list containing only the dicts that passed every check.
        """
        results = self.check_batch(examples, schema=schema)
        passed = []
        for ex, result in zip(examples, results):
            if result.passed:
                passed.append(ex)
            else:
                logger.debug("Filtered out: %s", result.summary())
        logger.info("Quality filter: %d / %d examples passed", len(passed), len(examples))
        return passed

    def _heuristic_checks(self, nl: str, sql: str, issues: list[str]) -> bool:
        """Run fast rule-based heuristic checks on an NL-SQL pair.

        Args:
            nl: Natural language question string.
            sql: SQL query string.
            issues: Mutable list to which human-readable issue descriptions
                are appended for any failing check.

        Returns:
            ``True`` if all heuristic checks pass, ``False`` otherwise.
        """
        ok = True

        # Check natural language
        if len(nl) < self.min_nl_length:
            issues.append(f"NL too short ({len(nl)} < {self.min_nl_length})")
            ok = False
        if len(nl) > self.max_nl_length:
            issues.append(f"NL too long ({len(nl)} > {self.max_nl_length})")
            ok = False

        # Check SQL
        if len(sql) < self.min_sql_length:
            issues.append(f"SQL too short ({len(sql)} < {self.min_sql_length})")
            ok = False
        if len(sql) > self.max_sql_length:
            issues.append(f"SQL too long ({len(sql)} > {self.max_sql_length})")
            ok = False

        # SQL should contain at least one SQL keyword
        sql_upper = sql.upper()
        has_keyword = any(kw in sql_upper for kw in SQL_KEYWORDS)
        if not has_keyword:
            issues.append("SQL missing standard keywords (SELECT/INSERT/UPDATE/DELETE/WITH)")
            ok = False

        # NL should not look like raw SQL
        nl_upper = nl.upper().strip()
        if nl_upper.startswith(("SELECT ", "INSERT ", "UPDATE ", "DELETE ", "WITH ")):
            issues.append("NL looks like raw SQL, not natural language")
            ok = False

        # Check for placeholder/template artifacts
        if "{{" in nl or "}}" in nl or "<PLACEHOLDER>" in nl.upper():
            issues.append("NL contains template placeholders")
            ok = False

        # SQL should have balanced parentheses
        if sql.count("(") != sql.count(")"):
            issues.append("SQL has unbalanced parentheses")
            ok = False

        return ok

    def _syntax_check(self, sql: str, issues: list[str]) -> bool:
        """Validate SQL syntax using sqlparse.

        sqlparse is lenient -- it parses almost anything -- so we check for
        specific indicators of broken SQL beyond just parsing.

        Args:
            sql: SQL query string to validate.
            issues: Mutable list to which human-readable issue descriptions
                are appended for any failing check.

        Returns:
            ``True`` if the SQL appears syntactically valid, ``False``
            otherwise.
        """
        try:
            parsed = sqlparse.parse(sql)
        except Exception:
            issues.append("sqlparse failed to parse SQL")
            return False

        if not parsed:
            issues.append("sqlparse returned empty result")
            return False

        statement = parsed[0]

        # Check that sqlparse identified a statement type
        stmt_type = statement.get_type()
        if stmt_type is None:
            # sqlparse returns None for unrecognizable SQL
            # But it also returns None for CTEs (WITH ... SELECT), which are valid
            sql_stripped = sql.strip().upper()
            if not sql_stripped.startswith("WITH"):
                issues.append("sqlparse could not determine statement type")
                return False

        # Check for common syntax errors that sqlparse won't catch
        sql_stripped = sql.strip()

        # Unclosed string literals
        single_quotes = sql_stripped.count("'")
        if single_quotes % 2 != 0:
            # Could be escaped quotes; do a more careful check
            unescaped = re.sub(r"''", "", sql_stripped)  # remove escaped quotes
            if unescaped.count("'") % 2 != 0:
                issues.append("SQL has unclosed string literal")
                return False

        # Statement should end cleanly (with ; or just the query)
        # No dangling commas at the end
        cleaned = sql_stripped.rstrip(";").rstrip()
        if cleaned.endswith(","):
            issues.append("SQL ends with a trailing comma")
            return False

        return True

    def _llm_score(self, nl: str, sql: str, schema: str | None = None) -> float:
        """Ask the teacher model to rate quality on a 1-5 scale.

        Scoring rubric:
        1 - Wrong or nonsensical
        2 - SQL has clear errors or doesn't match the question
        3 - Functional but awkward phrasing or suboptimal SQL
        4 - Good quality, minor improvements possible
        5 - Excellent -- natural question, correct & clean SQL

        Args:
            nl: Natural language question string.
            sql: SQL query string.
            schema: Optional SQL schema for additional context.

        Returns:
            A float score between 1.0 and 5.0. Returns 3.0 on failure so
            that a scoring error does not block the example.
        """
        schema_block = ""
        if schema:
            schema_block = f"\n\nDatabase schema:\n```sql\n{schema}\n```\n"

        messages = [
            Message(
                role="system",
                content=(
                    "You are a data quality evaluator for NL-to-SQL training pairs. "
                    "Rate the quality of the given pair on a 1-5 scale. "
                    "Respond with ONLY a JSON object: "
                    '{{"score": <1-5>, "reason": "<brief explanation>"}}'
                ),
            ),
            Message(
                role="user",
                content=(
                    f"Rate this NL->SQL training pair:{schema_block}\n\n"
                    f"Natural language: {nl}\n"
                    f"SQL: {sql}\n\n"
                    f"Scoring rubric:\n"
                    f"1 - Wrong or nonsensical\n"
                    f"2 - SQL has clear errors or doesn't match the question\n"
                    f"3 - Functional but awkward or suboptimal\n"
                    f"4 - Good quality, minor improvements possible\n"
                    f"5 - Excellent, natural question with correct clean SQL\n\n"
                    f'Return JSON: {{"score": <1-5>, "reason": "..."}}'
                ),
            ),
        ]

        try:
            response = self.client.complete(messages, temperature=0.0)
            return self._parse_score(response)
        except Exception:
            logger.exception("LLM quality scoring failed")
            # On failure, don't block the example — return a neutral score
            return 3.0

    def _parse_score(self, response: str) -> float:
        """Extract numeric score from LLM response.

        Args:
            response: Raw text response from the teacher model, expected to
                contain a JSON object with a 'score' key or a bare integer.

        Returns:
            A float score clamped to the 1.0-5.0 range. Defaults to 3.0
            when parsing fails.
        """
        text = response.strip()
        # Remove code fences
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)

        # Try JSON parsing
        try:
            data = json.loads(text)
            score = float(data.get("score", 3))
            return max(1.0, min(5.0, score))
        except (json.JSONDecodeError, TypeError, ValueError):
            pass

        # Fallback: look for a bare number
        match = re.search(r"\b([1-5])\b", text)
        if match:
            return float(match.group(1))

        logger.warning("Could not parse LLM score from response: %s", text[:100])
        return 3.0
