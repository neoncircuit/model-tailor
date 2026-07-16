"""Execution correctness gate for SQL generation.

Provides a deterministic oracle for NL-to-SQL quality: materialise the task
schema into a transient SQLite database, execute a predicted SQL query, and
classify the outcome. When a reference query is supplied, the gate also
executes it and compares unordered result sets.

This is the SQL-generation equivalent of the deterministic gate described in
the LinkedIn post: a small, non-ML component that rejects any output whose
claims (here, query results) cannot be verified against source evidence (the
database schema).
"""

from __future__ import annotations

import concurrent.futures
import logging
import sqlite3
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

logger = logging.getLogger(__name__)

DEFAULT_TIMEOUT = 5.0
DEFAULT_UNSAFE_KEYWORDS = {
    "DROP",
    "ALTER",
    "TRUNCATE",
    "PRAGMA",
    "ATTACH",
    "DETACH",
    "VACUUM",
}

# Statements that mutate data. Allowed by default in generation mode (so
# INSERT/UPDATE/DELETE examples can be validated), but can be blocked.
WRITE_KEYWORDS = {"INSERT", "UPDATE", "DELETE"}


@dataclass
class ExecutionResult:
    """Outcome of executing a single SQL query through the gate.

    Attributes:
        passed: True if the query satisfied the gate criteria.
        status: Machine-readable status label. One of:
            ``passed``, ``syntax_error``, ``execution_error``, ``wrong_result``,
            ``timeout``, ``empty_result``, ``unsafe``.
        error_message: Human-readable explanation when ``passed`` is False.
        pred_result: Row tuples returned by the predicted query, if any.
        target_result: Row tuples returned by the reference query, if provided.
    """

    passed: bool
    status: str
    error_message: str | None = None
    pred_result: list[tuple] | None = None
    target_result: list[tuple] | None = None

    def summary(self) -> str:
        """Return a compact, human-readable summary of this result."""
        if self.passed:
            return "[PASS]"
        parts = [f"[FAIL:{self.status}]"]
        if self.error_message:
            parts.append(self.error_message)
        return " ".join(parts)


@dataclass
class GateMetrics:
    """Aggregate counts produced by running the gate over a batch.

    Attributes:
        total: Total number of examples checked.
        passed_first_try: Examples that passed the gate on the first attempt.
        failed_first_try: Examples that failed the gate on the first attempt.
        repair_attempts: Total repair attempts made (if a repairer was used).
        repair_success: Repairs that resulted in a passing gate check.
        repair_failed: Repairs that still failed after max attempts.
        final_passed: Examples passing after any repairs.
        final_failed: Examples still failing after any repairs.
        failure_by_type: Mapping from status label to count.
    """

    total: int = 0
    passed_first_try: int = 0
    failed_first_try: int = 0
    repair_attempts: int = 0
    repair_success: int = 0
    repair_failed: int = 0
    final_passed: int = 0
    final_failed: int = 0
    failure_by_type: dict[str, int] = field(default_factory=dict)

    def to_flat_dict(self) -> dict[str, float]:
        """Convert metrics to flat floats suitable for MLFlow logging."""
        result: dict[str, float] = {
            "gate/total": float(self.total),
            "gate/passed_first_try": float(self.passed_first_try),
            "gate/failed_first_try": float(self.failed_first_try),
            "gate/repair_attempts": float(self.repair_attempts),
            "gate/repair_success": float(self.repair_success),
            "gate/repair_failed": float(self.repair_failed),
            "gate/final_passed": float(self.final_passed),
            "gate/final_failed": float(self.final_failed),
        }
        if self.total > 0:
            result["gate/pass_rate_first_try"] = self.passed_first_try / self.total
            result["gate/pass_rate_final"] = self.final_passed / self.total
            result["gate/repair_success_rate"] = (
                self.repair_success / self.repair_attempts if self.repair_attempts > 0 else 0.0
            )
        for status, count in self.failure_by_type.items():
            result[f"gate/failure_{status}"] = float(count)
        return result


class ExecutionGate:
    """Deterministic execution gate for SQL queries.

    The gate materialises a fresh SQLite database from a provided DDL schema,
    executes the predicted SQL, and classifies the result. A reference query
    can optionally be supplied for result-set comparison.

    Each ``check()`` call uses a clean database to avoid state leakage between
    queries. Timeouts are enforced at the Python level via a thread-pool so
    that runaway queries cannot hang the process.

    Args:
        schema_sql_path: Path to a ``.sql`` file containing DDL statements.
        db_path: Optional path to a persistent SQLite database. If omitted, a
            temporary file-backed database is created per ``ExecutionGate``
            instance and reused across ``check()`` calls within that instance.
        timeout: Maximum seconds allowed for a single query execution.
        allow_empty_result: If False, queries returning zero rows are treated
            as failures with status ``empty_result``.
        allow_write: If False, INSERT/UPDATE/DELETE queries are rejected.
        unsafe_keywords: Set of uppercase keywords that cause a query to be
            rejected with status ``unsafe`` before execution.
    """

    def __init__(
        self,
        schema_sql_path: str | None = None,
        db_path: str | None = None,
        timeout: float = DEFAULT_TIMEOUT,
        allow_empty_result: bool = True,
        allow_write: bool = True,
        unsafe_keywords: set[str] | None = None,
    ) -> None:
        """Initialize the gate.

        Args:
            schema_sql_path: Path to DDL file. May be omitted if a pre-built
                *db_path* is supplied.
            db_path: Optional persistent SQLite database path.
            timeout: Per-query timeout in seconds.
            allow_empty_result: Whether zero-row results are acceptable.
            allow_write: Whether DML statements are permitted.
            unsafe_keywords: Keywords that mark a query as unsafe. Defaults to
                DROP, ALTER, TRUNCATE, PRAGMA, ATTACH, DETACH, VACUUM.
        """
        self.schema_sql_path = schema_sql_path
        self.timeout = timeout
        self.allow_empty_result = allow_empty_result
        self.allow_write = allow_write
        self.unsafe_keywords = unsafe_keywords or DEFAULT_UNSAFE_KEYWORDS.copy()

        self._schema_sql: str | None = None
        if schema_sql_path:
            schema_path = Path(schema_sql_path)
            if schema_path.exists():
                self._schema_sql = schema_path.read_text(encoding="utf-8")
            else:
                logger.warning("Schema file not found: %s", schema_sql_path)

        if db_path:
            self._db_path = db_path
            self._owns_db = False
        else:
            self._db_file = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
            self._db_path = self._db_file.name
            self._owns_db = True

        if self._schema_sql:
            self._materialise_db()

    def _materialise_db(self) -> None:
        """Create tables in the backing database from the DDL file."""
        if not self._schema_sql:
            return
        conn = sqlite3.connect(self._db_path, timeout=self.timeout)
        try:
            conn.executescript(self._schema_sql)
        except sqlite3.Error as exc:
            logger.error("Failed to materialise schema: %s", exc)
            raise
        finally:
            conn.close()

    def _check_safety(self, sql: str) -> ExecutionResult | None:
        """Return an unsafe result if the query contains forbidden keywords."""
        upper = sql.upper()
        tokens = upper.split()
        for keyword in self.unsafe_keywords:
            if keyword in tokens:
                return ExecutionResult(
                    passed=False,
                    status="unsafe",
                    error_message=f"Query contains unsafe keyword: {keyword}",
                )
        if not self.allow_write:
            for keyword in WRITE_KEYWORDS:
                if keyword in tokens:
                    return ExecutionResult(
                        passed=False,
                        status="unsafe",
                        error_message=f"Write statements are not allowed: {keyword}",
                    )
        return None

    def _execute_with_timeout(
        self,
        sql: str,
    ) -> tuple[list[tuple] | None, str | None]:
        """Execute a query with a timeout and return rows or an error string.

        A fresh connection is opened inside the worker thread to avoid SQLite
        thread-safety restrictions.
        """

        def _run() -> list[tuple]:
            conn = sqlite3.connect(self._db_path, timeout=self.timeout)
            try:
                cursor = conn.execute(sql)
                return cursor.fetchall()
            finally:
                conn.close()

        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(_run)
                rows = future.result(timeout=self.timeout)
                return rows, None
        except concurrent.futures.TimeoutError:
            return None, f"Query timed out after {self.timeout}s"
        except sqlite3.Error as exc:
            return None, f"SQLite error: {exc}"
        except Exception as exc:
            return None, f"Execution error: {exc}"

    def _check_single(
        self,
        pred_sql: str,
        target_sql: str | None = None,
    ) -> ExecutionResult:
        """Run the gate for one predicted SQL query.

        Args:
            pred_sql: The generated SQL query to validate.
            target_sql: Optional reference query for result-set comparison.

        Returns:
            ExecutionResult with pass/fail status and details.
        """
        if not pred_sql or not pred_sql.strip():
            return ExecutionResult(
                passed=False,
                status="syntax_error",
                error_message="Empty SQL query",
            )

        # Safety check
        unsafe_result = self._check_safety(pred_sql)
        if unsafe_result:
            return unsafe_result

        try:
            conn = sqlite3.connect(self._db_path, timeout=self.timeout)
        except sqlite3.Error as exc:
            return ExecutionResult(
                passed=False,
                status="execution_error",
                error_message=f"Could not connect to database: {exc}",
            )

        try:
            pred_rows, pred_error = self._execute_with_timeout(pred_sql)
            if pred_error is not None:
                status = "timeout" if "timed out" in pred_error else "execution_error"
                return ExecutionResult(
                    passed=False,
                    status=status,
                    error_message=pred_error,
                )

            if not self.allow_empty_result and pred_rows is not None and len(pred_rows) == 0:
                return ExecutionResult(
                    passed=False,
                    status="empty_result",
                    error_message="Query returned no rows",
                    pred_result=[],
                )

            # If no target query, passing execution is enough
            if target_sql is None:
                return ExecutionResult(
                    passed=True,
                    status="passed",
                    pred_result=pred_rows,
                )

            # Safety check on target
            target_unsafe = self._check_safety(target_sql)
            if target_unsafe:
                return ExecutionResult(
                    passed=False,
                    status=target_unsafe.status,
                    error_message=f"Target query unsafe: {target_unsafe.error_message}",
                    pred_result=pred_rows,
                )

            target_rows, target_error = self._execute_with_timeout(target_sql)
            if target_error is not None:
                return ExecutionResult(
                    passed=False,
                    status="execution_error",
                    error_message=f"Target query failed: {target_error}",
                    pred_result=pred_rows,
                )

            if set(pred_rows or []) == set(target_rows or []):
                return ExecutionResult(
                    passed=True,
                    status="passed",
                    pred_result=pred_rows,
                    target_result=target_rows,
                )

            return ExecutionResult(
                passed=False,
                status="wrong_result",
                error_message="Predicted result set does not match target result set",
                pred_result=pred_rows,
                target_result=target_rows,
            )
        finally:
            conn.close()

    def check(
        self,
        pred_sql: str,
        target_sql: str | None = None,
    ) -> ExecutionResult:
        """Execute ``pred_sql`` against the schema and classify the outcome.

        Args:
            pred_sql: Generated SQL query.
            target_sql: Optional gold/reference SQL query for result comparison.

        Returns:
            ExecutionResult describing pass/fail status and details.
        """
        return self._check_single(pred_sql, target_sql)

    def check_batch(
        self,
        records: list[dict],
        pred_key: str = "sql",
        target_key: str = "target_sql",
    ) -> list[ExecutionResult]:
        """Run the gate over a batch of records.

        Args:
            records: Dicts containing at least the predicted SQL under *pred_key*.
            pred_key: Field name for the predicted SQL.
            target_key: Field name for the optional reference SQL. Records missing
                this key run in target-less mode.

        Returns:
            A list of ExecutionResult objects in the same order as *records*.
        """
        results: list[ExecutionResult] = []
        for record in records:
            pred = record.get(pred_key, "")
            target = record.get(target_key)
            results.append(self.check(pred, target))
        return results

    def filter_passed(
        self,
        records: list[dict],
        pred_key: str = "sql",
        target_key: str = "target_sql",
    ) -> list[dict]:
        """Return only records whose SQL passes the gate.

        Args:
            records: Dicts containing predicted SQL.
            pred_key: Field name for predicted SQL.
            target_key: Field name for optional reference SQL.

        Returns:
            Filtered list of records that passed the gate.
        """
        results = self.check_batch(records, pred_key, target_key)
        passed = []
        for record, result in zip(records, results):
            if result.passed:
                passed.append(record)
            else:
                logger.debug("Gate filtered out: %s", result.summary())
        logger.info("Execution gate: %d / %d passed", len(passed), len(records))
        return passed

    def compute_metrics(
        self,
        results: list[ExecutionResult],
    ) -> GateMetrics:
        """Aggregate gate results into metrics.

        Args:
            results: List of ExecutionResult objects.

        Returns:
            GateMetrics with counts and pass rates.
        """
        metrics = GateMetrics(total=len(results))
        for result in results:
            metrics.failure_by_type[result.status] = (
                metrics.failure_by_type.get(result.status, 0) + 1
            )
            if result.passed:
                metrics.passed_first_try += 1
                metrics.final_passed += 1
            else:
                metrics.failed_first_try += 1
                metrics.final_failed += 1
        return metrics

    def close(self) -> None:
        """Clean up any transient database created by this gate."""
        if self._owns_db and hasattr(self, "_db_file"):
            try:
                self._db_file.close()
                Path(self._db_file.name).unlink(missing_ok=True)
            except Exception as exc:
                logger.warning("Failed to clean up temp DB: %s", exc)

    def __del__(self) -> None:
        """Attempt to clean up the temporary database on garbage collection."""
        self.close()


# Keep a backwards-compatible callable for metrics.py and existing tests.
def _execution_gate_factory(
    schema_sql_path: str | None = None,
    db_path: str | None = None,
    timeout: float = DEFAULT_TIMEOUT,
    allow_empty_result: bool = True,
) -> Callable[[str, str], ExecutionResult]:
    """Create a closure that checks SQL against a schema.

    This factory exists so legacy callers can obtain a simple
    ``(pred, target) -> ExecutionResult`` function without managing the
    ``ExecutionGate`` lifecycle.
    """
    gate = ExecutionGate(
        schema_sql_path=schema_sql_path,
        db_path=db_path,
        timeout=timeout,
        allow_empty_result=allow_empty_result,
    )
    return gate.check
