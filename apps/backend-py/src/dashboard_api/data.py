"""Dashboard data loaders.

Pure data-access helpers for the local model-tailor dashboard. These functions
avoid framework-specific imports so they stay easy to unit test and reusable.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

logger = logging.getLogger(__name__)


class MlflowLoader:
    """Load runs and metric histories from a local MLFlow tracking server.

    Args:
        tracking_uri: URI of the MLFlow tracking server.
        experiment_name: Name of the experiment to query.
    """

    def __init__(self, tracking_uri: str, experiment_name: str) -> None:
        """Initialize the loader with tracking URI and experiment name.

        Args:
            tracking_uri: URI of the MLFlow tracking server.
            experiment_name: Name of the experiment to query.
        """
        self.tracking_uri = tracking_uri
        self.experiment_name = experiment_name

    def list_runs(self) -> pd.DataFrame:
        """Return a DataFrame summarising every run in the experiment.

        Columns include ``run_id``, ``status``, ``start_time``, ``params``,
        and ``metrics``.

        Returns:
            DataFrame with one row per run, or an empty DataFrame when the
            experiment cannot be found or the server is unreachable.
        """
        try:
            import mlflow
            from mlflow.tracking import MlflowClient
        except ImportError:
            logger.warning("mlflow is not installed.")
            return pd.DataFrame()

        try:
            mlflow.set_tracking_uri(self.tracking_uri)
            client = MlflowClient()
            experiment = client.get_experiment_by_name(self.experiment_name)
            if experiment is None:
                logger.warning(
                    "MLFlow experiment '%s' not found at %s",
                    self.experiment_name,
                    self.tracking_uri,
                )
                return pd.DataFrame()

            runs = client.search_runs(experiment_ids=[experiment.experiment_id])
        except Exception as exc:  # pragma: no cover - network/server failures
            logger.warning("Could not query MLFlow server: %s", exc)
            return pd.DataFrame()

        rows: list[dict[str, Any]] = []
        for run in runs:
            info = run.info
            rows.append(
                {
                    "run_id": info.run_id,
                    "status": info.status,
                    "start_time": pd.to_datetime(info.start_time, unit="ms"),
                    "params": dict(run.data.params),
                    "metrics": dict(run.data.metrics),
                }
            )

        return pd.DataFrame(rows)

    def get_metric_history(self, run_id: str, metric: str) -> pd.DataFrame:
        """Fetch the time-series for a single metric in a run.

        Args:
            run_id: MLFlow run identifier.
            metric: Metric key to retrieve (e.g. ``train/loss``).

        Returns:
            DataFrame with ``step`` and ``value`` columns, or an empty
            DataFrame if the run/metric cannot be loaded.
        """
        try:
            import mlflow
            from mlflow.tracking import MlflowClient
        except ImportError:
            return pd.DataFrame()

        try:
            mlflow.set_tracking_uri(self.tracking_uri)
            client = MlflowClient()
            history = client.get_metric_history(run_id, metric)
        except Exception as exc:  # pragma: no cover
            logger.debug("Could not load metric history for %s/%s: %s", run_id, metric, exc)
            return pd.DataFrame()

        rows = [{"step": m.step, "value": m.value} for m in history]
        return pd.DataFrame(rows)


def load_benchmark_result(path: Path) -> dict[str, Any]:
    """Load a single benchmark report JSON file.

    Args:
        path: Path to the JSON report.

    Returns:
        Benchmark report dictionary with added ``path`` and ``filename`` keys.
    """
    data = json.loads(path.read_text(encoding="utf-8"))
    data["path"] = str(path)
    data["filename"] = path.name
    return data


def scan_benchmark_results(directory: Path) -> list[dict[str, Any]]:
    """Load all benchmark JSON reports in a directory.

    Args:
        directory: Directory to scan for ``*.json`` files.

    Returns:
        List of benchmark report dictionaries, sorted by filename.
    """
    if not directory.exists():
        return []

    results: list[dict[str, Any]] = []
    for path in sorted(directory.glob("*.json")):
        try:
            results.append(load_benchmark_result(path))
        except Exception as exc:  # pragma: no cover - corrupted files
            logger.warning("Skipping unreadable benchmark file %s: %s", path, exc)
    return results


def load_yaml_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML configuration file.

    Args:
        path: Path to the YAML file.

    Returns:
        Parsed YAML content, or an empty dict if the file cannot be read.
    """
    try:
        with open(path, encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except Exception as exc:  # pragma: no cover
        logger.warning("Could not load config %s: %s", path, exc)
        return {}


def read_mlflow_config(config_path: str | Path = "config/base.yaml") -> dict[str, str]:
    """Read MLFlow tracking settings from the base config.

    Args:
        config_path: Path to ``base.yaml``.

    Returns:
        Dictionary with ``tracking_uri`` and ``experiment_name``.
    """
    cfg = load_yaml_config(config_path)
    mlflow_cfg = cfg.get("mlflow", {})
    return {
        "tracking_uri": mlflow_cfg.get(
            "tracking_uri",
            os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"),
        ),
        "experiment_name": mlflow_cfg.get("experiment_name", "model-tailor"),
    }
