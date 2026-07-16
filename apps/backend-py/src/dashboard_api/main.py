"""FastAPI backend for the local model-tailor dashboard."""

from __future__ import annotations

import asyncio
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pandas import isna

from dashboard_api import models
from dashboard_api.data import (
    MlflowLoader,
    load_yaml_config,
    read_mlflow_config,
    scan_benchmark_results,
)
from dashboard_api.system import snapshot_system

logger = logging.getLogger(__name__)

# Guard the synchronous system snapshot so it cannot block the event loop for
# more than a few seconds, and so overlapping frontend polls do not pile up.
_SYSTEM_SNAPSHOT_TIMEOUT_SECONDS = 5.0
_system_snapshot_lock = asyncio.Lock()

# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------


def _find_project_root() -> Path:
    """Locate the model-tailor project root from this file's location.

    Walks up from ``apps/backend-py/src/dashboard_api`` until it finds a
    directory containing ``config/base.yaml``.

    Returns:
        Absolute path to the project root.

    Raises:
        RuntimeError: If the project root cannot be located.
    """
    marker = Path("config/base.yaml")
    current = Path(__file__).resolve().parent
    for parent in [current] + list(current.parents):
        if (parent / marker).exists():
            return parent
    raise RuntimeError("Could not locate model-tailor project root.")


_PROJECT_ROOT = _find_project_root()


def _benchmark_results_dir() -> Path:
    """Return the directory that holds benchmark JSON reports."""
    env_dir = os.getenv("BENCHMARK_RESULTS_DIR")
    if env_dir:
        return Path(env_dir).resolve()
    return _PROJECT_ROOT / "tasks" / "sql_generation" / "results"


# ---------------------------------------------------------------------------
# Application factory
# ---------------------------------------------------------------------------


def create_app() -> FastAPI:
    """Create and return a configured FastAPI application.

    Returns:
        FastAPI app with dashboard endpoints.
    """

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        mlflow_cfg = read_mlflow_config(_PROJECT_ROOT / "config" / "base.yaml")
        app.state.mlflow_cfg = mlflow_cfg
        app.state.loader = MlflowLoader(
            tracking_uri=mlflow_cfg["tracking_uri"],
            experiment_name=mlflow_cfg["experiment_name"],
        )
        app.state.benchmark_dir = _benchmark_results_dir()
        app.state.project_root = _PROJECT_ROOT
        yield

    application = FastAPI(
        title="model-tailor dashboard API",
        version="0.1.0",
        description="Backend for the local model-tailor configuration and performance dashboard.",
        lifespan=lifespan,
    )

    application.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    _register_routes(application)
    return application


def _register_routes(application: FastAPI) -> None:
    """Attach endpoint handlers to *application*."""

    @application.get("/health", response_model=models.HealthResponse, tags=["system"])
    async def health_check() -> models.HealthResponse:
        """Return API health status."""
        return models.HealthResponse(status="ok")

    @application.get("/config", response_model=models.MlflowConfigResponse, tags=["mlflow"])
    async def get_config() -> models.MlflowConfigResponse:
        """Return MLFlow tracking configuration."""
        cfg = application.state.mlflow_cfg
        return models.MlflowConfigResponse(
            tracking_uri=cfg["tracking_uri"],
            experiment_name=cfg["experiment_name"],
        )

    @application.get("/runs", response_model=list[models.RunSummary], tags=["mlflow"])
    async def list_runs() -> list[models.RunSummary]:
        """List MLFlow runs for the configured experiment."""
        loader: MlflowLoader = application.state.loader
        df = loader.list_runs()
        if df.empty:
            return []

        rows: list[models.RunSummary] = []
        for _, row in df.iterrows():
            start = row["start_time"]
            start_dt = start.to_pydatetime() if start is not None and not isna(start) else None
            rows.append(
                models.RunSummary(
                    run_id=str(row["run_id"]),
                    status=str(row["status"]),
                    start_time=start_dt,
                    params={str(k): str(v) for k, v in row["params"].items()},
                    metrics={str(k): float(v) for k, v in row["metrics"].items()},
                )
            )
        return rows

    @application.get(
        "/runs/{run_id}/metrics/{metric_key:path}",
        response_model=models.MetricSeries,
        tags=["mlflow"],
    )
    async def get_metric_history(run_id: str, metric_key: str) -> models.MetricSeries:
        """Fetch a metric time-series for a run."""
        loader: MlflowLoader = application.state.loader
        df = loader.get_metric_history(run_id, metric_key)
        values = [
            models.MetricPoint(step=int(row["step"]), value=float(row["value"]))
            for _, row in df.iterrows()
        ]
        return models.MetricSeries(run_id=run_id, metric=metric_key, values=values)

    @application.get(
        "/benchmarks",
        response_model=list[models.BenchmarkSummary],
        tags=["benchmarks"],
    )
    async def list_benchmarks() -> list[models.BenchmarkSummary]:
        """List available benchmark JSON reports."""
        reports = scan_benchmark_results(application.state.benchmark_dir)
        summaries: list[models.BenchmarkSummary] = []
        for report in reports:
            summaries.append(
                models.BenchmarkSummary(
                    filename=report["filename"],
                    model=report.get("model"),
                    num_examples=report.get("num_examples"),
                    elapsed_seconds=report.get("elapsed_seconds"),
                    overall_metrics=report.get("overall_metrics", {}),
                )
            )
        return summaries

    @application.get(
        "/benchmarks/{filename}",
        response_model=models.BenchmarkReport,
        tags=["benchmarks"],
    )
    async def get_benchmark(filename: str) -> models.BenchmarkReport:
        """Return a single benchmark report."""
        if not filename.endswith(".json") or "/" in filename or ".." in filename:
            raise HTTPException(status_code=400, detail="Invalid filename")

        path = application.state.benchmark_dir / filename
        try:
            path.resolve().relative_to(application.state.benchmark_dir.resolve())
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid filename") from None

        if not path.exists():
            raise HTTPException(status_code=404, detail="Benchmark not found")

        from dashboard_api.data import load_benchmark_result

        report = load_benchmark_result(path)
        return models.BenchmarkReport(
            filename=report.pop("filename"),
            report=report,
        )

    @application.get("/system", response_model=models.SystemSnapshot, tags=["system"])
    async def get_system_snapshot() -> models.SystemSnapshot:
        """Return a live snapshot of host CPU/RAM/GPU/storage status."""
        async with _system_snapshot_lock:
            try:
                snapshot = await asyncio.wait_for(
                    asyncio.to_thread(snapshot_system),
                    timeout=_SYSTEM_SNAPSHOT_TIMEOUT_SECONDS,
                )
            except asyncio.TimeoutError as exc:
                logger.warning(
                    "System snapshot timed out after %ss",
                    _SYSTEM_SNAPSHOT_TIMEOUT_SECONDS,
                )
                raise HTTPException(
                    status_code=503,
                    detail="System snapshot timed out",
                ) from exc
        return models.SystemSnapshot(**snapshot)

    @application.get("/configs", response_model=models.ConfigFiles, tags=["config"])
    async def get_configs() -> models.ConfigFiles:
        """Return base and task YAML configs."""
        base_path = application.state.project_root / "config" / "base.yaml"
        task_path = application.state.project_root / "config" / "tasks" / "sql_generation.yaml"
        return models.ConfigFiles(
            base=load_yaml_config(base_path),
            task=load_yaml_config(task_path),
        )
