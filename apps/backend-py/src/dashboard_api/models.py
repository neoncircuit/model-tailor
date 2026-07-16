"""Pydantic response schemas for the dashboard API."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    """Response from the ``/health`` endpoint."""

    status: str


class MlflowConfigResponse(BaseModel):
    """Response from the ``/config`` endpoint."""

    tracking_uri: str
    experiment_name: str


class RunSummary(BaseModel):
    """Summary of a single MLFlow run."""

    run_id: str
    status: str
    start_time: datetime | None = None
    params: dict[str, str]
    metrics: dict[str, float]


class MetricPoint(BaseModel):
    """A single ``(step, value)`` observation for a metric."""

    step: int
    value: float


class MetricSeries(BaseModel):
    """Time-series response for a run metric."""

    run_id: str
    metric: str
    values: list[MetricPoint]


class BenchmarkSummary(BaseModel):
    """Short summary of a benchmark report."""

    filename: str
    model: str | None = None
    num_examples: int | None = None
    elapsed_seconds: float | None = None
    overall_metrics: dict[str, Any] = Field(default_factory=dict)


class BenchmarkReport(BaseModel):
    """Full benchmark report loaded from a JSON file."""

    filename: str
    report: dict[str, Any]


class GpuDevice(BaseModel):
    """GPU device snapshot."""

    index: int
    name: str
    utilisation_pct: float | None
    memory_used_gb: float
    memory_total_gb: float
    memory_percent: float


class GpuSnapshot(BaseModel):
    """GPU section of a system snapshot."""

    available: bool
    devices: list[GpuDevice]


class RamSnapshot(BaseModel):
    """RAM section of a system snapshot."""

    used_gb: float
    total_gb: float
    percent: float


class StorageDrive(BaseModel):
    """Storage drive / partition snapshot."""

    mountpoint: str
    device: str
    fstype: str
    total_gb: float
    used_gb: float
    free_gb: float
    percent: float
    builtin: bool
    bus_type: str | None = None


class SystemSnapshot(BaseModel):
    """Response from the ``/system`` endpoint."""

    timestamp: float
    cpu_percent: float
    ram: RamSnapshot
    gpu: GpuSnapshot
    storage: list[StorageDrive] = Field(default_factory=list)


class ConfigFiles(BaseModel):
    """Response from the ``/configs`` endpoint."""

    base: dict[str, Any]
    task: dict[str, Any]
