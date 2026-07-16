"""Tests for the dashboard FastAPI backend."""

from __future__ import annotations

import json
import time
from pathlib import Path
from unittest.mock import MagicMock

import pandas as pd
import pytest
from fastapi.testclient import TestClient

import dashboard_api.main as main_module
from dashboard_api.main import create_app


@pytest.fixture()
def app(tmp_path: Path):
    """Create an app with mocked state for testing."""
    application = create_app()
    application.state.mlflow_cfg = {
        "tracking_uri": "http://localhost:5000",
        "experiment_name": "test-experiment",
    }

    loader = MagicMock()
    loader.list_runs.return_value = pd.DataFrame(
        [
            {
                "run_id": "run-1",
                "status": "FINISHED",
                "start_time": pd.Timestamp("2026-07-16T10:00:00"),
                "params": {"lr": "0.001"},
                "metrics": {"train/loss": 0.5},
            }
        ]
    )
    loader.get_metric_history.return_value = pd.DataFrame(
        [{"step": 0, "value": 0.5}, {"step": 1, "value": 0.4}]
    )
    application.state.loader = loader
    application.state.benchmark_dir = tmp_path / "benchmarks"
    application.state.benchmark_dir.mkdir()
    application.state.project_root = tmp_path

    # Create minimal config files used by /configs.
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    (config_dir / "base.yaml").write_text("project: test\n")
    task_dir = config_dir / "tasks"
    task_dir.mkdir()
    (task_dir / "sql_generation.yaml").write_text("task: sql\n")

    return application


@pytest.fixture()
def client(app):
    """Return a TestClient for the mocked app."""
    return TestClient(app)


def test_health(client: TestClient):
    """``/health`` should return ok."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_config(client: TestClient):
    """``/config`` should return MLFlow settings."""
    response = client.get("/config")
    assert response.status_code == 200
    assert response.json() == {
        "tracking_uri": "http://localhost:5000",
        "experiment_name": "test-experiment",
    }


def test_list_runs(client: TestClient):
    """``/runs`` should return run summaries."""
    response = client.get("/runs")
    assert response.status_code == 200
    payload = response.json()
    assert len(payload) == 1
    assert payload[0]["run_id"] == "run-1"
    assert payload[0]["metrics"]["train/loss"] == 0.5


def test_metric_history(client: TestClient):
    """``/runs/{id}/metrics/{key}`` should return time-series points."""
    response = client.get("/runs/run-1/metrics/train/loss")
    assert response.status_code == 200
    payload = response.json()
    assert payload["run_id"] == "run-1"
    assert payload["metric"] == "train/loss"
    assert len(payload["values"]) == 2
    assert payload["values"][1] == {"step": 1, "value": 0.4}


def test_list_benchmarks(client: TestClient):
    """``/benchmarks`` should list available reports."""
    report = {
        "model": "test-model",
        "num_examples": 10,
        "elapsed_seconds": 5.0,
        "overall_metrics": {"accuracy": 0.9},
    }
    benchmark_path = client.app.state.benchmark_dir / "test-model_20260716_100000.json"
    benchmark_path.write_text(json.dumps(report), encoding="utf-8")

    response = client.get("/benchmarks")
    assert response.status_code == 200
    payload = response.json()
    assert len(payload) == 1
    assert payload[0]["filename"] == "test-model_20260716_100000.json"
    assert payload[0]["model"] == "test-model"


def test_get_benchmark(client: TestClient):
    """``/benchmarks/{filename}`` should return a full report."""
    report = {"model": "test-model", "num_examples": 10}
    benchmark_path = client.app.state.benchmark_dir / "report.json"
    benchmark_path.write_text(json.dumps(report), encoding="utf-8")

    response = client.get("/benchmarks/report.json")
    assert response.status_code == 200
    payload = response.json()
    assert payload["filename"] == "report.json"
    assert payload["report"]["model"] == "test-model"


def test_get_benchmark_invalid_filename(client: TestClient):
    """Invalid filenames should be rejected."""
    response = client.get("/benchmarks/not-a-json.txt")
    assert response.status_code == 400


def test_system(client: TestClient, monkeypatch):
    """``/system`` should return a live system snapshot."""
    snapshot = {
        "timestamp": 1234567890.0,
        "cpu_percent": 12.5,
        "ram": {"used_gb": 8.0, "total_gb": 16.0, "percent": 50.0},
        "gpu": {"available": False, "devices": []},
        "storage": [],
    }
    monkeypatch.setattr(main_module, "snapshot_system", lambda: snapshot)

    response = client.get("/system")
    assert response.status_code == 200
    payload = response.json()
    assert payload["cpu_percent"] == 12.5
    assert payload["ram"]["total_gb"] == 16.0
    assert payload["gpu"]["available"] is False
    assert "storage" in payload


def test_configs(client: TestClient):
    """``/configs`` should return base and task configs."""
    response = client.get("/configs")
    assert response.status_code == 200
    payload = response.json()
    assert payload["base"]["project"] == "test"
    assert payload["task"]["task"] == "sql"


def test_system_snapshot_timeout_returns_503(client: TestClient, monkeypatch):
    """``/system`` should return 503 when the snapshot times out."""
    def _slow_snapshot():
        time.sleep(main_module._SYSTEM_SNAPSHOT_TIMEOUT_SECONDS + 1)
        return {}

    monkeypatch.setattr(main_module, "snapshot_system", _slow_snapshot)

    response = client.get("/system")
    assert response.status_code == 503
    assert "timed out" in response.json()["detail"].lower()
