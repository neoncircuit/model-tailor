"""Tests for the storage scanner and built-in/external classification."""

from __future__ import annotations

import json
import time
from collections import namedtuple
from unittest.mock import MagicMock, patch

import pytest

from dashboard_api import storage

Partition = namedtuple("Partition", ["device", "mountpoint", "fstype"])


@pytest.fixture(autouse=True)
def _clear_caches():
    """Reset module-level caches before each test."""
    storage._powershell_cache.clear()
    storage._storage_snapshot_cache = None
    storage._lsblk_transport_cache.clear()
    yield
    storage._powershell_cache.clear()
    storage._storage_snapshot_cache = None
    storage._lsblk_transport_cache.clear()


def _make_partition(device="/dev/sda1", mountpoint="/", fstype="ext4"):
    """Return a psutil-style partition namedtuple."""
    return Partition(device=device, mountpoint=mountpoint, fstype=fstype)


def _make_usage(total, used, free=None):
    """Return a shutil-style disk_usage namedtuple."""
    return namedtuple("Usage", ["total", "used", "free"])(
        total=total,
        used=used,
        free=free if free is not None else total - used,
    )


def test_filters_pseudo_filesystems():
    """Pseudo filesystems should be skipped."""
    partitions = [
        _make_partition("/dev/sda1", "/", "ext4"),
        _make_partition("none", "/tmp", "tmpfs"),
        _make_partition("proc", "/proc", "proc"),
    ]
    with (
        patch.object(storage.psutil, "disk_partitions", return_value=partitions),
        patch.object(storage.shutil, "disk_usage", return_value=_make_usage(100, 50)),
        patch.object(storage, "_classify_partition", return_value=(True, None)),
    ):
        result = storage.snapshot_storage()

    assert len(result) == 1
    assert result[0]["mountpoint"] == "/"


def test_filters_zero_total_partitions():
    """Partitions reporting zero total capacity should be skipped."""
    partitions = [
        _make_partition("/dev/sda1", "/", "ext4"),
        _make_partition("/dev/sdb1", "/data", "ext4"),
    ]

    def usage(path):
        if path == "/data":
            return _make_usage(0, 0)
        return _make_usage(100, 50)

    with (
        patch.object(storage.psutil, "disk_partitions", return_value=partitions),
        patch.object(storage.shutil, "disk_usage", side_effect=usage),
        patch.object(storage, "_classify_partition", return_value=(True, None)),
    ):
        result = storage.snapshot_storage()

    assert len(result) == 1
    assert result[0]["mountpoint"] == "/"


def test_computes_gb_and_percent():
    """Capacity and usage should be converted to GB and rounded."""
    partitions = [_make_partition("/dev/sda1", "/", "ext4")]
    usage = _make_usage(200 * 1024**3, 50 * 1024**3, 150 * 1024**3)
    with (
        patch.object(storage.psutil, "disk_partitions", return_value=partitions),
        patch.object(storage.shutil, "disk_usage", return_value=usage),
        patch.object(storage, "_classify_partition", return_value=(True, "SATA")),
    ):
        result = storage.snapshot_storage()

    assert len(result) == 1
    drive = result[0]
    assert drive["total_gb"] == 200.0
    assert drive["used_gb"] == 50.0
    assert drive["free_gb"] == 150.0
    assert drive["percent"] == 25.0
    assert drive["builtin"] is True
    assert drive["bus_type"] == "SATA"


def test_classify_wsl_windows_drive_builtin():
    """WSL drvfs drive with a built-in bus type should be marked built-in."""
    part = _make_partition("drvfs", "/mnt/c", "9p")
    mapping = {"C": "NVMe"}
    with (
        patch.object(storage, "_query_windows_bus_mapping", return_value=mapping),
        patch.object(storage.platform, "release", return_value="5.15.0-microsoft-standard"),
    ):
        builtin, bus_type = storage._classify_partition(part)

    assert builtin is True
    assert bus_type == "NVME"


def test_classify_wsl_windows_drive_external():
    """WSL drvfs drive attached via USB should be marked external."""
    part = _make_partition("drvfs", "/mnt/d", "9p")
    mapping = {"D": "USB"}
    with (
        patch.object(storage, "_query_windows_bus_mapping", return_value=mapping),
        patch.object(storage.platform, "release", return_value="5.15.0-microsoft-standard"),
    ):
        builtin, bus_type = storage._classify_partition(part)

    assert builtin is False
    assert bus_type == "USB"


def test_classify_wsl_internal_builtin():
    """WSL internal rootfs partitions should be treated as built-in."""
    part = _make_partition("/dev/sdb", "/", "ext4")
    with patch.object(storage.platform, "release", return_value="5.15.0-microsoft-standard"):
        builtin, bus_type = storage._classify_partition(part)

    assert builtin is True
    assert bus_type == "WSL"


def test_classify_linux_removable():
    """Linux partitions backed by a removable block device are external."""
    part = _make_partition("/dev/sda1", "/media/usb", "vfat")
    with (
        patch.object(storage.platform, "release", return_value="5.15.0-generic"),
        patch.object(storage.platform, "system", return_value="Linux"),
        patch.object(storage.Path, "read_text", return_value="1"),
        patch.object(storage, "_lsblk_transport", return_value=None),
    ):
        builtin, bus_type = storage._classify_partition(part)

    assert builtin is False
    assert bus_type == "Removable"


def test_classify_linux_usb_transport():
    """Linux partitions with lsblk TRAN=usb are external."""
    part = _make_partition("/dev/sda1", "/media/usb", "vfat")
    with (
        patch.object(storage.platform, "release", return_value="5.15.0-generic"),
        patch.object(storage.platform, "system", return_value="Linux"),
        patch.object(storage.Path, "read_text", return_value="0"),
        patch.object(storage, "_lsblk_transport", return_value="usb"),
    ):
        builtin, bus_type = storage._classify_partition(part)

    assert builtin is False
    assert bus_type == "USB"


def test_classify_linux_sata_transport():
    """Linux partitions with lsblk TRAN=sata are built-in."""
    part = _make_partition("/dev/sda1", "/", "ext4")
    with (
        patch.object(storage.platform, "release", return_value="5.15.0-generic"),
        patch.object(storage.platform, "system", return_value="Linux"),
        patch.object(storage.Path, "read_text", return_value="0"),
        patch.object(storage, "_lsblk_transport", return_value="sata"),
    ):
        builtin, bus_type = storage._classify_partition(part)

    assert builtin is True
    assert bus_type == "SATA"


def test_base_block_device():
    """Base block-device extraction should strip partition suffixes."""
    assert storage._base_block_device("sda1") == "sda"
    assert storage._base_block_device("nvme0n1p1") == "nvme0n1"
    assert storage._base_block_device("mmcblk0p2") == "mmcblk0"


def test_query_windows_bus_mapping_parses_json():
    """PowerShell JSON output should be parsed into a drive-letter mapping."""
    ps_output = json.dumps(
        [
            {"DriveLetter": "C", "BusType": "NVMe"},
            {"DriveLetter": "D", "BusType": "USB"},
        ]
    )
    completed = MagicMock()
    completed.returncode = 0
    completed.stdout = ps_output
    completed.stderr = ""
    with patch.object(storage.subprocess, "run", return_value=completed):
        mapping = storage._query_windows_bus_mapping()

    assert mapping == {"C": "NVMe", "D": "USB"}


def test_query_windows_bus_mapping_failure_returns_empty():
    """A PowerShell failure should return an empty mapping instead of crashing."""
    completed = MagicMock()
    completed.returncode = 1
    completed.stdout = ""
    completed.stderr = "Access denied"
    with patch.object(storage.subprocess, "run", return_value=completed):
        mapping = storage._query_windows_bus_mapping()

    assert mapping == {}


def test_powershell_cache_uses_cached_value():
    """Cached PowerShell results should be reused within the TTL."""
    mapping = {"C": "NVMe"}
    storage._powershell_cache["mapping"] = (mapping, time.monotonic())
    with patch.object(storage, "_query_windows_bus_mapping") as query:
        result = storage._cached_windows_bus_mapping()
        query.assert_not_called()
    assert result == mapping
    storage._powershell_cache.clear()


def test_unknown_platform_defaults_external():
    """Unknown platforms should classify drives as external to stay safe."""
    part = _make_partition("/dev/disk0s1", "/", "apfs")
    with (
        patch.object(storage.platform, "system", return_value="Darwin"),
        patch.object(storage.platform, "release", return_value="22.0"),
    ):
        builtin, bus_type = storage._classify_partition(part)

    assert builtin is False
    assert bus_type is None


def test_snapshot_storage_cache_uses_cached_value():
    """A fresh cache should be returned without touching psutil or shutil."""
    cached = [{"mountpoint": "/cached"}]
    storage._storage_snapshot_cache = (cached, time.monotonic())
    with (
        patch.object(storage.psutil, "disk_partitions") as partitions,
        patch.object(storage.shutil, "disk_usage") as usage,
    ):
        result = storage.snapshot_storage()

    partitions.assert_not_called()
    usage.assert_not_called()
    assert result is cached


def test_snapshot_storage_cache_expires():
    """A stale cache should trigger a fresh scan."""
    cached = [{"mountpoint": "/cached"}]
    storage._storage_snapshot_cache = (
        cached,
        time.monotonic() - storage._STORAGE_SNAPSHOT_CACHE_TTL_SECONDS - 1,
    )
    partitions = [_make_partition("/dev/sda1", "/", "ext4")]
    with (
        patch.object(storage.psutil, "disk_partitions", return_value=partitions),
        patch.object(storage, "_disk_usage_safe", return_value=_make_usage(100, 50)),
        patch.object(storage, "_classify_partition", return_value=(True, "SATA")),
    ):
        result = storage.snapshot_storage()

    assert len(result) == 1
    assert result[0]["mountpoint"] == "/"


def test_lsblk_transport_cache_uses_cached_value():
    """Repeated lsblk lookups for the same device should use the cache."""
    storage._lsblk_transport_cache["sda"] = ("sata", time.monotonic())
    with patch.object(storage.subprocess, "run") as run:
        result = storage._lsblk_transport("sda")
        run.assert_not_called()
    assert result == "sata"


def test_lsblk_transport_cache_expires():
    """A stale lsblk cache should trigger a fresh subprocess call."""
    storage._lsblk_transport_cache["sda"] = (
        "sata",
        time.monotonic() - storage._LSBLK_CACHE_TTL_SECONDS - 1,
    )
    completed = MagicMock()
    completed.stdout = "usb\n"
    with patch.object(storage.subprocess, "run", return_value=completed) as run:
        result = storage._lsblk_transport("sda")

    run.assert_called_once()
    assert result == "usb"


def test_disk_usage_timeout_skips_partition():
    """A disk_usage timeout should skip the partition instead of crashing."""
    partitions = [
        _make_partition("/dev/sda1", "/", "ext4"),
        _make_partition("/dev/sdb1", "/data", "ext4"),
    ]
    future = MagicMock()
    future.result.side_effect = [TimeoutError("stale mount"), _make_usage(100, 50)]
    with (
        patch.object(storage.psutil, "disk_partitions", return_value=partitions),
        patch.object(storage._disk_usage_executor, "submit", return_value=future),
        patch.object(storage, "_classify_partition", return_value=(True, "SATA")),
    ):
        result = storage.snapshot_storage()

    assert len(result) == 1
    assert result[0]["mountpoint"] == "/data"
