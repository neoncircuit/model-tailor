"""Tests for src/evaluate/gate.py.

All tests run without API keys, GPU, or external services.
"""

from __future__ import annotations

import pytest

from src.evaluate.gate import ExecutionGate, ExecutionResult


@pytest.fixture
def schema_path(tmp_path):
    """Create a small SQLite schema file for testing."""
    path = tmp_path / "test_schema.sql"
    path.write_text(
        """
        CREATE TABLE users (
            id INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            age INTEGER
        );
        INSERT INTO users (id, name, age) VALUES (1, 'Alice', 30);
        INSERT INTO users (id, name, age) VALUES (2, 'Bob', 25);
        INSERT INTO users (id, name, age) VALUES (3, 'Carol', 35);

        CREATE TABLE orders (
            id INTEGER PRIMARY KEY,
            user_id INTEGER,
            amount REAL
        );
        INSERT INTO orders (id, user_id, amount) VALUES (1, 1, 99.99);
        INSERT INTO orders (id, user_id, amount) VALUES (2, 1, 50.00);
        INSERT INTO orders (id, user_id, amount) VALUES (3, 2, 75.00);
        """
    )
    return str(path)


@pytest.fixture
def gate(schema_path):
    """Return an ExecutionGate wired to the test schema."""
    g = ExecutionGate(schema_sql_path=schema_path, timeout=2.0)
    yield g
    g.close()


class TestExecutionGateCheck:
    """Tests for ExecutionGate.check()."""

    def test_correct_query_passes_with_target(self, gate):
        """A query returning the same result set as the target passes."""
        result = gate.check(
            "SELECT name FROM users WHERE age > 25 ORDER BY name",
            "SELECT name FROM users WHERE age > 25 ORDER BY name",
        )
        assert result.passed is True
        assert result.status == "passed"
        assert {"Alice", "Carol"} == {row[0] for row in (result.pred_result or [])}

    def test_correct_query_passes_without_target(self, gate):
        """A valid query with no target passes the gate."""
        result = gate.check("SELECT COUNT(*) FROM users")
        assert result.passed is True
        assert result.status == "passed"
        assert result.pred_result == [(3,)]

    def test_wrong_result_fails(self, gate):
        """A query returning a different result set than the target fails."""
        result = gate.check(
            "SELECT name FROM users WHERE age > 30",
            "SELECT name FROM users WHERE age > 25",
        )
        assert result.passed is False
        assert result.status == "wrong_result"
        assert result.pred_result == [("Carol",)]

    def test_bad_table_execution_error(self, gate):
        """A query referencing a non-existent table returns an execution error."""
        result = gate.check("SELECT * FROM nonexistent_table")
        assert result.passed is False
        assert result.status == "execution_error"
        assert "no such table" in (result.error_message or "").lower()

    def test_bad_column_execution_error(self, gate):
        """A query referencing a non-existent column returns an execution error."""
        result = gate.check("SELECT nonexistent_column FROM users")
        assert result.passed is False
        assert result.status == "execution_error"

    def test_empty_result_allowed_by_default(self, gate):
        """Zero-row results pass when allow_empty_result is True."""
        result = gate.check("SELECT * FROM users WHERE age > 100")
        assert result.passed is True
        assert result.pred_result == []

    def test_empty_result_rejected_when_configured(self, schema_path):
        """Zero-row results fail when allow_empty_result is False."""
        strict_gate = ExecutionGate(
            schema_sql_path=schema_path,
            allow_empty_result=False,
            timeout=2.0,
        )
        try:
            result = strict_gate.check("SELECT * FROM users WHERE age > 100")
            assert result.passed is False
            assert result.status == "empty_result"
        finally:
            strict_gate.close()

    def test_unsafe_keyword_rejected(self, gate):
        """Queries containing unsafe keywords are rejected before execution."""
        result = gate.check("DROP TABLE users")
        assert result.passed is False
        assert result.status == "unsafe"
        assert "DROP" in (result.error_message or "")

    def test_write_statements_blocked_when_disabled(self, schema_path):
        """INSERT/UPDATE/DELETE are rejected when allow_write is False."""
        strict_gate = ExecutionGate(
            schema_sql_path=schema_path,
            allow_write=False,
            timeout=2.0,
        )
        try:
            result = strict_gate.check("INSERT INTO users (name, age) VALUES ('Dave', 40)")
            assert result.passed is False
            assert result.status == "unsafe"
        finally:
            strict_gate.close()

    def test_empty_sql_fails(self, gate):
        """An empty SQL string is treated as a syntax error."""
        result = gate.check("")
        assert result.passed is False
        assert result.status == "syntax_error"

    def test_reference_query_can_also_fail(self, gate):
        """If the target query fails, the gate reports an execution error."""
        result = gate.check(
            "SELECT name FROM users WHERE age > 25",
            "SELECT name FROM nonexistent_table",
        )
        assert result.passed is False
        assert result.status == "execution_error"
        assert "target query failed" in (result.error_message or "").lower()


class TestExecutionGateBatch:
    """Tests for batch helpers."""

    def test_check_batch_preserves_order(self, gate):
        """check_batch returns results in the same order as the input records."""
        records = [
            {"sql": "SELECT COUNT(*) FROM users", "target_sql": "SELECT COUNT(*) FROM users"},
            {"sql": "SELECT * FRUM users", "target_sql": "SELECT * FROM users"},
            {"sql": "DROP TABLE users"},
        ]
        results = gate.check_batch(records)
        assert len(results) == 3
        assert results[0].passed is True
        assert results[1].passed is False
        assert results[2].passed is False
        assert results[2].status == "unsafe"

    def test_filter_passed_keeps_only_passing_records(self, gate):
        """filter_passed returns only records that pass the gate."""
        records = [
            {"sql": "SELECT COUNT(*) FROM users", "target_sql": "SELECT COUNT(*) FROM users"},
            {"sql": "SELECT * FRUM users"},
            {
                "sql": "SELECT name FROM users WHERE age > 100",
                "target_sql": "SELECT name FROM users WHERE age > 100",
            },
        ]
        passed = gate.filter_passed(records)
        assert len(passed) == 2
        assert passed[0]["sql"] == "SELECT COUNT(*) FROM users"
        assert passed[1]["sql"] == "SELECT name FROM users WHERE age > 100"


class TestGateMetrics:
    """Tests for GateMetrics aggregation."""

    def test_compute_metrics_counts_pass_and_fail(self, gate):
        """compute_metrics produces correct totals and failure breakdown."""
        results = [
            ExecutionResult(passed=True, status="passed"),
            ExecutionResult(passed=False, status="wrong_result"),
            ExecutionResult(passed=False, status="wrong_result"),
            ExecutionResult(passed=False, status="unsafe"),
        ]
        metrics = gate.compute_metrics(results)
        assert metrics.total == 4
        assert metrics.passed_first_try == 1
        assert metrics.failed_first_try == 3
        assert metrics.final_passed == 1
        assert metrics.final_failed == 3
        assert metrics.failure_by_type == {
            "wrong_result": 2,
            "unsafe": 1,
            "passed": 1,
        }

    def test_to_flat_dict_includes_rates(self, gate):
        """to_flat_dict exposes pass rates and per-status counts."""
        results = [
            ExecutionResult(passed=True, status="passed"),
            ExecutionResult(passed=False, status="wrong_result"),
        ]
        metrics = gate.compute_metrics(results)
        flat = metrics.to_flat_dict()
        assert flat["gate/total"] == 2.0
        assert flat["gate/pass_rate_first_try"] == 0.5
        assert flat["gate/pass_rate_final"] == 0.5
        assert flat["gate/failure_wrong_result"] == 1.0


class TestExecutionGateLifecycle:
    """Tests for resource cleanup."""

    def test_close_removes_temp_database(self, schema_path):
        """close() removes the temporary SQLite database created by the gate."""
        gate = ExecutionGate(schema_sql_path=schema_path, timeout=2.0)
        db_path = gate._db_path
        gate.close()
        assert not __import__("pathlib").Path(db_path).exists()

    def test_persistent_db_not_removed_on_close(self, tmp_path, schema_path):
        """close() does not delete a user-supplied persistent database."""
        db_path = tmp_path / "persist.db"
        gate = ExecutionGate(schema_sql_path=schema_path, db_path=str(db_path), timeout=2.0)
        gate.close()
        assert db_path.exists()
