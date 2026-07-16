"""Integration tests for gate + repair inside BatchGenerator.

These tests exercise ``BatchGenerator._run_gate_and_repair`` and the
configuration parser with real ``ExecutionGate`` instances but mocked
teacher clients, so no API keys or GPUs are required.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from src.generate.batch import BatchConfig, BatchGenerator
from src.generate.strategies import GeneratedExample


@pytest.fixture
def schema_path(tmp_path: Path) -> Path:
    """Return the path to a small SQLite schema used by the execution gate."""
    path = tmp_path / "schema.sql"
    path.write_text(
        "CREATE TABLE users (id INTEGER, name TEXT);\n"
        "INSERT INTO users VALUES (1, 'Alice'), (2, 'Bob');\n",
        encoding="utf-8",
    )
    return path


@pytest.fixture
def base_config(tmp_path: Path, schema_path: Path) -> BatchConfig:
    """Return a BatchConfig with generation, quality and gate/repair disabled."""
    return BatchConfig(
        num_examples=2,
        output_dir=str(tmp_path),
        output_filename="generated.jsonl",
        run_quality_check=False,
        run_execution_gate=False,
        repair_enabled=False,
    )


def test_gate_disabled_passes_everything(base_config: BatchConfig) -> None:
    """When the gate is disabled, every example is accepted unchanged."""
    examples = [
        GeneratedExample(natural_language="Q1", sql="SELECT * FROM users;"),
        GeneratedExample(natural_language="Q2", sql="SELECT * FROM missing_table;"),
    ]
    generator = BatchGenerator(MagicMock(), config=base_config)

    accepted, rejected, metrics = generator._run_gate_and_repair(examples)

    assert accepted == examples
    assert rejected == []
    assert metrics.total == 2


def test_gate_filters_invalid_sql(base_config: BatchConfig, schema_path: Path) -> None:
    """A failing execution gate rejects invalid SQL when repair is disabled."""
    base_config.run_execution_gate = True
    base_config.schema_file = str(schema_path)
    examples = [
        GeneratedExample(natural_language="Valid", sql="SELECT * FROM users;"),
        GeneratedExample(natural_language="Invalid", sql="SELECT * FROM missing_table;"),
    ]
    generator = BatchGenerator(MagicMock(), config=base_config)

    accepted, rejected, metrics = generator._run_gate_and_repair(examples)

    assert len(accepted) == 1
    assert len(rejected) == 1
    assert accepted[0].natural_language == "Valid"
    assert rejected[0].natural_language == "Invalid"
    assert rejected[0].metadata["gate_status"] == "execution_error"
    assert rejected[0].metadata["was_repaired"] is False
    assert metrics.passed_first_try == 1
    assert metrics.failed_first_try == 1
    assert metrics.final_passed == 1
    assert metrics.final_failed == 1


def test_repair_fixes_invalid_sql(base_config: BatchConfig, schema_path: Path) -> None:
    """A mocked teacher can repair invalid SQL so the example is accepted."""
    base_config.run_execution_gate = True
    base_config.repair_enabled = True
    base_config.repair_max_attempts = 2
    base_config.schema_file = str(schema_path)

    client = MagicMock()
    client.complete.return_value = "SELECT * FROM users;"

    examples = [
        GeneratedExample(natural_language="Broken", sql="SELECT * FROM missing_table;")
    ]
    generator = BatchGenerator(client, config=base_config)

    accepted, rejected, metrics = generator._run_gate_and_repair(examples)

    assert len(accepted) == 1
    assert rejected == []
    assert accepted[0].sql == "SELECT * FROM users;"
    assert accepted[0].metadata["was_repaired"] is True
    assert accepted[0].metadata["repair_attempts"] == 1
    assert accepted[0].metadata["original_sql"] == "SELECT * FROM missing_table;"
    assert metrics.repair_success == 1
    assert metrics.repair_failed == 0
    assert metrics.final_passed == 1


def test_repair_gives_up_after_max_attempts(base_config: BatchConfig, schema_path: Path) -> None:
    """If the repairer cannot fix the SQL, the example is rejected."""
    base_config.run_execution_gate = True
    base_config.repair_enabled = True
    base_config.repair_max_attempts = 2
    base_config.schema_file = str(schema_path)

    client = MagicMock()
    client.complete.return_value = "SELECT * FROM another_missing_table;"

    examples = [
        GeneratedExample(natural_language="Broken", sql="SELECT * FROM missing_table;")
    ]
    generator = BatchGenerator(client, config=base_config)

    accepted, rejected, metrics = generator._run_gate_and_repair(examples)

    assert accepted == []
    assert len(rejected) == 1
    assert rejected[0].metadata["was_repaired"] is True
    assert rejected[0].metadata["repair_attempts"] == 2
    assert metrics.repair_success == 0
    assert metrics.repair_failed == 1
    assert client.complete.call_count == 2


def test_run_saves_accepted_and_rejected(base_config: BatchConfig, schema_path: Path) -> None:
    """``run()`` persists accepted and rejected examples in separate JSONL files."""
    base_config.run_execution_gate = True
    base_config.schema_file = str(schema_path)
    base_config.num_examples = 2

    examples = [
        GeneratedExample(natural_language="Valid", sql="SELECT * FROM users;"),
        GeneratedExample(natural_language="Invalid", sql="SELECT * FROM missing_table;"),
    ]
    generator = BatchGenerator(MagicMock(), config=base_config)

    # Bypass async generation and inject examples directly.
    generator._plan_jobs = lambda: [object()]  # type: ignore[method-assign]

    async def _fake_run_job(job, sem, prog):  # type: ignore[no-redef]
        return examples

    generator._run_job = _fake_run_job  # type: ignore[method-assign]

    result = generator.run_sync()

    assert result.total_generated == 2
    assert result.total_passed_quality == 2
    assert result.total_passed_gate == 1
    assert result.total_rejected == 1

    assert result.output_path is not None
    accepted_lines = Path(result.output_path).read_text(encoding="utf-8").strip().split("\n")
    assert len(accepted_lines) == 1
    accepted_record = json.loads(accepted_lines[0])
    assert accepted_record["natural_language"] == "Valid"

    assert result.rejected_path is not None
    rejected_lines = Path(result.rejected_path).read_text(encoding="utf-8").strip().split("\n")
    assert len(rejected_lines) == 1
    rejected_record = json.loads(rejected_lines[0])
    assert rejected_record["natural_language"] == "Invalid"


def test_config_parses_gate_and_repair_sections(tmp_path: Path) -> None:
    """BatchConfig.from_yaml reads execution_gate and repair subsections."""
    config_path = tmp_path / "task.yaml"
    config_path.write_text(
        "generation:\n"
        "  num_examples: 10\n"
        "  execution_gate:\n"
        "    enabled: true\n"
        "    timeout: 3.0\n"
        "    allow_empty_result: false\n"
        "  repair:\n"
        "    enabled: true\n"
        "    max_attempts: 3\n"
        "    repair_batch_size: 7\n"
        "curation:\n"
        "  require_valid_sql: false\n",
        encoding="utf-8",
    )

    config = BatchConfig.from_yaml(str(config_path))

    assert config.num_examples == 10
    assert config.run_execution_gate is True
    assert config.execution_gate_timeout == 3.0
    assert config.execution_gate_allow_empty_result is False
    assert config.repair_enabled is True
    assert config.repair_max_attempts == 3
    assert config.repair_batch_size == 7
    assert config.run_quality_check is False
