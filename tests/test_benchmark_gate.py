"""Tests for gate/repair metrics inside BenchmarkRunner and MLFlow logging."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.evaluate.benchmark import BenchmarkRunner
from src.evaluate.gate import ExecutionGate
from src.generate.repair import SQLRepairer
from src.train.monitor import log_gate_metrics


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
def gate(schema_path: Path) -> ExecutionGate:
    """Return an ExecutionGate initialised with the toy schema."""
    return ExecutionGate(schema_sql_path=str(schema_path))


def _make_runner(gate: ExecutionGate, **kwargs: object) -> BenchmarkRunner:
    """Create a BenchmarkRunner configured with the given gate."""
    return BenchmarkRunner(
        batch_size=2,
        max_new_tokens=32,
        gate=gate,
        **kwargs,
    )


def test_gate_metrics_included_when_gate_provided(gate: ExecutionGate) -> None:
    """A provided gate adds pass/fail metrics to the benchmark results."""
    runner = _make_runner(gate)
    # ``sql`` is the gold/reference query; predictions are model outputs.
    test_data = [
        {"nl": "Get all users", "sql": "SELECT * FROM users;"},
        {"nl": "Get all users", "sql": "SELECT * FROM users;"},
    ]
    predictions = ["SELECT * FROM users;", "SELECT * FROM missing_table;"]

    with patch(
        "src.evaluate.benchmark._generate_predictions",
        return_value=predictions,
    ):
        results = runner.run(MagicMock(), MagicMock(), test_data)

    assert results.gate_metrics["gate/total"] == 2.0
    assert results.gate_metrics["gate/passed_first_try"] == 1.0
    assert results.gate_metrics["gate/failed_first_try"] == 1.0
    assert results.metric_scores["gate_pass_rate"] == 0.5
    assert results.metric_scores["gate_final_pass_rate"] == 0.5


def test_repair_improves_pass_rate(gate: ExecutionGate, schema_path: Path) -> None:
    """Using the repairer increases the final pass rate in the metrics."""
    client = MagicMock()
    client.complete.return_value = "SELECT * FROM users;"
    repairer = SQLRepairer(client=client, gate=gate, max_attempts=2)
    runner = _make_runner(gate, repairer=repairer, use_repair=True)

    test_data = [
        {"nl": "Get all users", "sql": "SELECT * FROM users;"},
        {"nl": "Get all users", "sql": "SELECT * FROM users;"},
    ]
    predictions = ["SELECT * FROM users;", "SELECT * FROM missing_table;"]

    with patch(
        "src.evaluate.benchmark._generate_predictions",
        return_value=predictions,
    ):
        results = runner.run(MagicMock(), MagicMock(), test_data)

    assert results.metric_scores["gate_pass_rate"] == 0.5
    assert results.metric_scores["writer_only_pass_rate"] == 0.5
    assert results.metric_scores["writer_plus_fixer_pass_rate"] == 1.0
    assert results.metric_scores["repair_success_rate"] == 1.0
    assert results.repair_metrics["repair/success"] == 1.0


def test_runner_validates_repair_requires_gate() -> None:
    """BenchmarkRunner raises when use_repair is True but no gate is supplied."""
    with pytest.raises(ValueError, match="use_repair requires an ExecutionGate"):
        BenchmarkRunner(use_repair=True)


def test_log_gate_metrics_logs_to_mlflow_and_wandb() -> None:
    """log_gate_metrics forwards metrics to MLFlow and WandB when available."""
    mlflow_module = MagicMock()
    mlflow_module.active_run.return_value = MagicMock()
    wandb_module = MagicMock()
    wandb_module.run = MagicMock()

    metrics = {"gate/pass_rate_first_try": 0.75, "gate/pass_rate_final": 0.9}

    with patch.dict(
        "sys.modules",
        {"mlflow": mlflow_module, "wandb": wandb_module},
    ):
        log_gate_metrics(metrics, step=7)

    mlflow_module.log_metrics.assert_called_once_with(metrics, step=7)
    wandb_module.log.assert_called_once_with(metrics)


def test_log_gate_metrics_is_noop_when_metrics_empty() -> None:
    """log_gate_metrics does nothing when given an empty dict."""
    mlflow_module = MagicMock()
    wandb_module = MagicMock()

    with patch.dict(
        "sys.modules",
        {"mlflow": mlflow_module, "wandb": wandb_module},
    ):
        log_gate_metrics({})

    mlflow_module.log_metrics.assert_not_called()
    wandb_module.log.assert_not_called()
