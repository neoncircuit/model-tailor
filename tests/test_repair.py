"""Tests for src/generate/repair.py.

All tests use mocked teacher and gate objects so they require no API keys,
GPU, or external services.
"""

from __future__ import annotations

import pytest
from unittest.mock import MagicMock

from src.evaluate.gate import ExecutionResult
from src.generate.repair import RepairResult, SQLRepairer


@pytest.fixture
def gate():
    """Return a mock ExecutionGate."""
    return MagicMock()


@pytest.fixture
def client():
    """Return a mock TeacherClient."""
    return MagicMock()


@pytest.fixture
def repairer(client, gate):
    """Return a SQLRepairer wired to mock dependencies."""
    return SQLRepairer(client=client, gate=gate, max_attempts=2)


class TestSQLRepairer:
    """Tests for the SQL repair loop."""

    def test_passes_initially_returns_original(self, client, gate, repairer):
        """When the original SQL passes the gate, no repair is attempted."""
        gate.check.return_value = ExecutionResult(passed=True, status="passed")
        record = {
            "natural_language": "Count users",
            "sql": "SELECT COUNT(*) FROM users",
        }

        result = repairer.repair(record)

        assert isinstance(result, RepairResult)
        assert result.attempts == 0
        assert result.repaired_sql == "SELECT COUNT(*) FROM users"
        assert result.original_sql == result.repaired_sql
        assert result.final_result.passed is True
        client.complete.assert_not_called()

    def test_repairs_on_first_attempt(self, client, gate, repairer):
        """A failing query that is fixed in one attempt returns the corrected SQL."""
        gate.check.side_effect = [
            ExecutionResult(
                passed=False,
                status="execution_error",
                error_message="no such column: emal",
            ),
            ExecutionResult(passed=True, status="passed"),
        ]
        client.complete.return_value = "SELECT email FROM users"
        record = {
            "natural_language": "List user emails",
            "sql": "SELECT emal FROM users",
        }

        result = repairer.repair(record)

        assert result.attempts == 1
        assert result.repaired_sql == "SELECT email FROM users"
        assert result.final_result.passed is True
        assert client.complete.call_count == 1

    def test_gives_up_after_max_attempts(self, client, gate, repairer):
        """If all repair attempts fail, the final failing result is returned."""
        gate.check.side_effect = [
            ExecutionResult(passed=False, status="execution_error"),
            ExecutionResult(passed=False, status="execution_error"),
            ExecutionResult(passed=False, status="execution_error"),
        ]
        client.complete.return_value = "SELECT * FROM nowhere"
        record = {
            "natural_language": "Bad query",
            "sql": "SELECT * FROM missing_table",
        }

        result = repairer.repair(record)

        assert result.attempts == 2
        assert result.final_result.passed is False
        assert result.final_result.status == "execution_error"
        assert client.complete.call_count == 2

    def test_stops_early_when_repair_passes(self, client, gate, repairer):
        """No further teacher calls are made once a repair passes."""
        gate.check.side_effect = [
            ExecutionResult(passed=False, status="wrong_result"),
            ExecutionResult(passed=True, status="passed"),
        ]
        client.complete.return_value = "SELECT name FROM users"
        record = {
            "natural_language": "Get names",
            "sql": "SELECT wrong_col FROM users",
        }

        repairer.repair(record)

        assert client.complete.call_count == 1
        assert gate.check.call_count == 2

    def test_extracts_sql_from_markdown_fence(self, client, gate, repairer):
        """Markdown code fences are stripped from the model response."""
        gate.check.side_effect = [
            ExecutionResult(passed=False, status="execution_error"),
            ExecutionResult(passed=True, status="passed"),
        ]
        client.complete.return_value = "```sql\nSELECT id FROM users\n```"
        record = {
            "natural_language": "Get ids",
            "sql": "SELECT ids FROM users",
        }

        result = repairer.repair(record)

        assert result.repaired_sql == "SELECT id FROM users"

    def test_prompt_contains_failure_feedback(self, client, gate):
        """The repair prompt includes the failure status, message and schema."""
        repairer = SQLRepairer(client=client, gate=gate, max_attempts=1)
        gate.check.return_value = ExecutionResult(
            passed=False,
            status="execution_error",
            error_message="no such column: emal",
        )
        client.complete.return_value = "SELECT email FROM users"
        record = {
            "natural_language": "List user emails",
            "sql": "SELECT emal FROM users",
        }
        schema = "CREATE TABLE users (email TEXT)"

        repairer.repair(record, schema=schema)

        messages = client.complete.call_args[0][0]
        prompt = messages[1].content
        assert "List user emails" in prompt
        assert "SELECT emal FROM users" in prompt
        assert "execution_error" in prompt
        assert "no such column: emal" in prompt
        assert schema in prompt

    def test_target_sql_used_by_gate_not_prompt(self, client, gate, repairer):
        """A reference query is passed to the gate but not leaked to the teacher."""
        gate.check.side_effect = [
            ExecutionResult(passed=False, status="wrong_result"),
            ExecutionResult(passed=True, status="passed"),
        ]
        client.complete.return_value = "SELECT name FROM users"
        record = {
            "natural_language": "Get names",
            "sql": "SELECT nme FROM users",
            "target_sql": "SELECT name FROM users",
        }

        result = repairer.repair(record)

        assert result.final_result.passed is True
        # Gate got target_sql on every check.
        for call in gate.check.call_args_list:
            assert call.args[1] == "SELECT name FROM users"
        # Teacher prompt must not contain the reference answer.
        messages = client.complete.call_args[0][0]
        prompt = messages[1].content
        assert "SELECT name FROM users" not in prompt

    def test_handles_teacher_exception(self, client, gate, repairer):
        """A teacher call exception is captured and does not propagate."""
        gate.check.return_value = ExecutionResult(
            passed=False,
            status="execution_error",
        )
        client.complete.side_effect = RuntimeError("API error")
        record = {
            "natural_language": "Bad query",
            "sql": "SELECT * FROM missing_table",
        }

        result = repairer.repair(record)

        assert result.final_result.passed is False
        assert any(
            entry["status"] == "teacher_error" for entry in result.history
        )

    def test_history_records_every_attempt(self, client, gate, repairer):
        """History contains an entry for the initial check and each attempt."""
        gate.check.side_effect = [
            ExecutionResult(passed=False, status="execution_error"),
            ExecutionResult(passed=True, status="passed"),
        ]
        client.complete.return_value = "SELECT id FROM users"
        record = {
            "natural_language": "Get ids",
            "sql": "SELECT ids FROM users",
        }

        result = repairer.repair(record)

        assert len(result.history) == 2
        assert result.history[0]["action"] == "initial_check"
        assert result.history[1]["action"] == "repair_attempt"
        assert result.history[1]["attempt"] == 1
