"""Live system snapshots for the dashboard."""

from __future__ import annotations

import logging
import time
from typing import Any

import psutil

logger = logging.getLogger(__name__)


def snapshot_system() -> dict[str, Any]:
    """Capture a single snapshot of host RAM, CPU, and GPU status.

    GPU information is collected via ``pynvml`` when available, falling back
    to ``torch.cuda`` if the dashboard host has CUDA but no ``pynvml``.
    When neither is available the GPU section reports ``available`` as
    ``False`` so the dashboard can render CPU-only metrics.

    Returns:
        Dictionary with ``timestamp``, ``cpu_percent``, ``ram`` totals,
        ``gpu`` status, and a ``storage`` list of drives.
    """
    from dashboard_api.storage import snapshot_storage

    ram = psutil.virtual_memory()
    snapshot: dict[str, Any] = {
        "timestamp": time.time(),
        "cpu_percent": psutil.cpu_percent(interval=None),
        "ram": {
            "used_gb": round(ram.used / 1024**3, 2),
            "total_gb": round(ram.total / 1024**3, 2),
            "percent": ram.percent,
        },
        "gpu": {"available": False, "devices": []},
        "storage": snapshot_storage(),
    }

    try:
        from pynvml import (
            nvmlDeviceGetCount,
            nvmlDeviceGetHandleByIndex,
            nvmlDeviceGetMemoryInfo,
            nvmlDeviceGetName,
            nvmlDeviceGetUtilizationRates,
            nvmlInit,
            nvmlShutdown,
        )

        nvmlInit()
        device_count = nvmlDeviceGetCount()
        snapshot["gpu"]["available"] = device_count > 0
        for i in range(device_count):
            handle = nvmlDeviceGetHandleByIndex(i)
            mem = nvmlDeviceGetMemoryInfo(handle)
            util = nvmlDeviceGetUtilizationRates(handle)
            name = nvmlDeviceGetName(handle)
            snapshot["gpu"]["devices"].append(
                {
                    "index": i,
                    "name": name,
                    "utilisation_pct": float(util.gpu),
                    "memory_used_gb": round(mem.used / 1024**3, 2),
                    "memory_total_gb": round(mem.total / 1024**3, 2),
                    "memory_percent": round(100.0 * mem.used / mem.total, 1),
                }
            )
        nvmlShutdown()
        return snapshot
    except Exception:
        logger.debug("pynvml unavailable or failed; trying torch.cuda fallback.", exc_info=True)

    try:
        import torch

        if torch.cuda.is_available():
            device = torch.cuda.current_device()
            snapshot["gpu"]["available"] = True
            total = torch.cuda.get_device_properties(device).total_memory
            allocated = torch.cuda.memory_allocated(device)
            snapshot["gpu"]["devices"].append(
                {
                    "index": device,
                    "name": torch.cuda.get_device_name(device),
                    "utilisation_pct": None,
                    "memory_used_gb": round(allocated / 1024**3, 2),
                    "memory_total_gb": round(total / 1024**3, 2),
                    "memory_percent": round(100.0 * allocated / total, 1),
                }
            )
    except Exception:
        logger.debug("torch.cuda unavailable; returning CPU-only snapshot.", exc_info=True)

    return snapshot
