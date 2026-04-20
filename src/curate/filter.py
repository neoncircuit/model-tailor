"""Quality filtering for curated datasets.

Provides a chainable ``QualityFilter`` that progressively narrows a record
list through length checks, SQL syntax validation, keyword requirements, and
score thresholds.
"""

from __future__ import annotations

import logging
from typing import Sequence

import sqlparse

logger = logging.getLogger(__name__)

Record = dict[str, object]

# Default SQL keywords that a well-formed training example should contain.
DEFAULT_SQL_KEYWORDS: set[str] = {"SELECT", "FROM"}


class QualityFilter:
    """Chainable quality filter over a list of data records.

    Usage::

        filtered = (
            QualityFilter(data)
            .filter_length(min_len=10, max_len=500)
            .filter_sql_valid()
            .filter_has_keywords()
            .filter_score(min_score=3.0)
            .result()
        )

    Every ``filter_*`` method returns ``self`` so calls can be chained.
    Call :meth:`result` to retrieve the final filtered list.

    Args:
        data: Input records. The original list is **not** mutated.
    """

    def __init__(self, data: list[Record]) -> None:
        # Work on a shallow copy so callers keep their original list intact.
        self._data: list[Record] = list(data)
        self._initial_count: int = len(data)

    # -- internal helpers -------------------------------------------------

    @staticmethod
    def _text_len(record: Record, field: str) -> int:
        """Return the character length of a text field in a record.

        Args:
            record: A data record dictionary.
            field: Key to look up in the record.

        Returns:
            Character length of the field value, or 0 if it is not a string.
        """
        value = record.get(field, "")
        return len(value) if isinstance(value, str) else 0

    def _apply(self, predicate, label: str) -> "QualityFilter":
        """Apply a predicate to filter the internal record list.

        Args:
            predicate: Callable that takes a record and returns True to keep it.
            label: Human-readable label used in log messages.

        Returns:
            Self, to allow method chaining.
        """
        before = len(self._data)
        self._data = [r for r in self._data if predicate(r)]
        removed = before - len(self._data)
        if removed:
            logger.info("QualityFilter.%s: removed %d / %d records", label, removed, before)
        return self

    # -- chainable filter methods -----------------------------------------

    def filter_length(
        self,
        min_len: int = 1,
        max_len: int = 10_000,
        fields: Sequence[str] = ("nl", "sql"),
    ) -> "QualityFilter":
        """Keep records where *every* listed field's length is in ``[min_len, max_len]``.

        Args:
            min_len: Minimum character length (inclusive).
            max_len: Maximum character length (inclusive).
            fields: Fields to check (default ``("nl", "sql")``).

        Returns:
            Self, to allow method chaining.
        """

        def _pred(rec: Record) -> bool:
            return all(min_len <= self._text_len(rec, f) <= max_len for f in fields)

        return self._apply(_pred, "filter_length")

    def filter_sql_valid(self, field: str = "sql") -> "QualityFilter":
        """Keep records whose *field* value parses as syntactically valid SQL.

        Uses ``sqlparse`` to tokenise the SQL string. A statement is
        considered invalid if:

        * it is empty / whitespace-only, or
        * ``sqlparse.parse`` returns no statements, or
        * the first token is of type ``Error``.

        Args:
            field: Record field containing the SQL string to validate.

        Returns:
            Self, to allow method chaining.
        """

        def _pred(rec: Record) -> bool:
            sql_text = rec.get(field, "")
            if not isinstance(sql_text, str) or not sql_text.strip():
                return False
            try:
                parsed = sqlparse.parse(sql_text)
            except Exception:
                return False
            if not parsed:
                return False
            # Check that the first statement's first meaningful token is not an error.
            tokens = [t for t in parsed[0].tokens if not t.is_whitespace]
            if not tokens:
                return False
            # sqlparse marks truly broken fragments with ttype Error.
            if tokens[0].ttype is sqlparse.tokens.Error:
                return False
            return True

        return self._apply(_pred, "filter_sql_valid")

    def filter_has_keywords(
        self,
        required: set[str] | None = None,
        field: str = "sql",
    ) -> "QualityFilter":
        """Keep records whose SQL contains all *required* keywords.

        Comparison is case-insensitive.

        Args:
            required: Keywords that must appear. Defaults to
                ``{"SELECT", "FROM"}``.
            field: Record field to inspect.

        Returns:
            Self, to allow method chaining.
        """
        kw = {k.upper() for k in (required or DEFAULT_SQL_KEYWORDS)}

        def _pred(rec: Record) -> bool:
            sql_text = rec.get(field, "")
            if not isinstance(sql_text, str):
                return False
            upper = sql_text.upper()
            return all(k in upper for k in kw)

        return self._apply(_pred, "filter_has_keywords")

    def filter_score(
        self,
        min_score: float = 3.0,
        field: str = "score",
    ) -> "QualityFilter":
        """Keep records whose numeric *field* value meets a minimum threshold.

        Records that lack the *field* entirely are **kept** (filter is
        lenient for optional metadata).

        Args:
            min_score: Minimum acceptable score (inclusive).
            field: Record field containing the score.

        Returns:
            Self, to allow method chaining.
        """

        def _pred(rec: Record) -> bool:
            val = rec.get(field)
            if val is None:
                # No score present -- allow through.
                return True
            try:
                return float(val) >= min_score
            except (TypeError, ValueError):
                return False

        return self._apply(_pred, "filter_score")

    # -- result -----------------------------------------------------------

    def result(self) -> list[Record]:
        """Return the filtered list of records.

        Returns:
            The records that survived all applied filters.
        """
        total_removed = self._initial_count - len(self._data)
        logger.info(
            "QualityFilter: %d -> %d records (%d removed in total)",
            self._initial_count,
            len(self._data),
            total_removed,
        )
        return self._data
