"""Storage drive discovery and built-in/external classification."""

from __future__ import annotations

import json
import logging
import platform
import re
import shutil
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import psutil

logger = logging.getLogger(__name__)

_PSEUDO_FILESYSTEMS = frozenset(
    {
        "tmpfs",
        "devtmpfs",
        "proc",
        "sysfs",
        "cgroup",
        "cgroup2",
        "devpts",
        "mqueue",
        "hugetlbfs",
        "tracefs",
        "debugfs",
        "fusectl",
        "fuse",
        "fuseblk",
        "overlay",
        "aufs",
        "squashfs",
        "efivarfs",
        "securityfs",
        "pstore",
        "bpf",
        "configfs",
        "binfmt_misc",
        "rpc_pipefs",
        "nfsd",
        "devfs",
        "kernfs",
        "none",
        "",
    }
)

_BUILTIN_BUS_TYPES = frozenset({"SATA", "NVME", "SAS", "ATA", "IDE", "SCSI"})
_EXTERNAL_BUS_TYPES = frozenset({"USB", "UASP"})

# Cache PowerShell bus-type lookups for ~30 seconds so the frontend poll does not
# spawn a shell twice a second.
_POWERSHELL_CACHE_TTL_SECONDS = 30
_powershell_cache: dict[str, tuple[dict[str, str], float]] = {}

# Cache the full storage snapshot so the 2-second frontend poll does not
# re-scan every partition on every tick.
_STORAGE_SNAPSHOT_CACHE_TTL_SECONDS = 5
_storage_snapshot_cache: tuple[list[dict[str, Any]], float] | None = None

# Cache lsblk transport lookups per base block device to avoid a subprocess
# storm when multiple partitions sit on the same disk.
_LSBLK_CACHE_TTL_SECONDS = 30
_lsblk_transport_cache: dict[str, tuple[str | None, float]] = {}

# Read disk usage in a thread pool with a per-call timeout so a stale mount
# cannot block the whole snapshot.
_DISK_USAGE_TIMEOUT_SECONDS = 1.0
_DISK_USAGE_MAX_WORKERS = 4
_disk_usage_executor = ThreadPoolExecutor(
    max_workers=_DISK_USAGE_MAX_WORKERS,
    thread_name_prefix="storage_disk_usage",
)


def _disk_usage_safe(mountpoint: str) -> Any | None:
    """Return disk usage for *mountpoint* with a short timeout.

    ``shutil.disk_usage()`` can hang on stale network or USB mounts. Running it
    in a thread pool lets us cap the wait time without blocking the caller.

    Args:
        mountpoint: Filesystem path to measure.

    Returns:
        A ``shutil`` disk-usage named tuple, or ``None`` if the read timed out
        or raised ``OSError``.
    """
    try:
        return _disk_usage_executor.submit(
            shutil.disk_usage, mountpoint
        ).result(timeout=_DISK_USAGE_TIMEOUT_SECONDS)
    except (TimeoutError, OSError) as exc:
        logger.debug("disk_usage failed for %s: %s", mountpoint, exc)
        return None


def snapshot_storage() -> list[dict[str, Any]]:
    """Return a snapshot of storage drives / partitions.

    Uses ``psutil.disk_partitions(all=True)`` because ``all=False`` hides WSL
    ``drvfs`` mounts. Pseudo filesystems and zero-capacity partitions are
    filtered out. Results are cached for ``_STORAGE_SNAPSHOT_CACHE_TTL_SECONDS``
    so frequent callers (e.g. the 2-second dashboard poll) do not re-scan the
    disks every time.

    Returns:
        List of drive dictionaries suitable for ``models.StorageDrive``.
    """
    global _storage_snapshot_cache

    now = time.monotonic()
    if _storage_snapshot_cache is not None:
        cached_value, cached_at = _storage_snapshot_cache
        if now - cached_at < _STORAGE_SNAPSHOT_CACHE_TTL_SECONDS:
            return cached_value

    drives: list[dict[str, Any]] = []
    for part in psutil.disk_partitions(all=True):
        if _skip_partition(part):
            continue
        usage = _disk_usage_safe(part.mountpoint)
        if usage is None or usage.total == 0:
            continue

        builtin, bus_type = _classify_partition(part)
        drives.append(
            {
                "mountpoint": part.mountpoint,
                "device": part.device,
                "fstype": part.fstype,
                "total_gb": round(usage.total / 1024**3, 2),
                "used_gb": round(usage.used / 1024**3, 2),
                "free_gb": round(usage.free / 1024**3, 2),
                "percent": round(100.0 * usage.used / usage.total, 1),
                "builtin": builtin,
                "bus_type": bus_type,
            }
        )

    _storage_snapshot_cache = (drives, now)
    return drives


def _skip_partition(part: psutil._psplatform.Partition) -> bool:
    """Return True for pseudo filesystems and loop/ram devices.

    Args:
        part: A ``psutil`` partition named tuple.

    Returns:
        Whether this partition should be excluded from the snapshot.
    """
    fstype = (part.fstype or "").lower()
    if fstype in _PSEUDO_FILESYSTEMS:
        return True
    device = part.device or ""
    return any(device.startswith(prefix) for prefix in ("/dev/loop", "/dev/ram", "/dev/zram"))


def _classify_partition(part: psutil._psplatform.Partition) -> tuple[bool, str | None]:
    """Classify a partition as built-in or external.

    Classification is platform aware:

    - WSL: map Windows drive letters to ``Get-PhysicalDisk BusType``.
    - Linux: inspect ``/sys/class/block/*/removable`` and ``lsblk TRAN``.
    - Windows: map logical drive letters to ``Get-PhysicalDisk BusType``.
    - Other: default to external to avoid silently mislabelling removable media.

    Args:
        part: A ``psutil`` partition named tuple.

    Returns:
        Tuple of ``(builtin, bus_type)``. ``bus_type`` may be ``None`` if it
        could not be determined.
    """
    system = platform.system().lower()
    is_wsl = "microsoft" in platform.release().lower()

    if is_wsl:
        return _classify_wsl(part)
    if system == "linux":
        return _classify_linux(part)
    if system == "windows":
        return _classify_windows(part)
    return False, None


def _classify_wsl(part: psutil._psplatform.Partition) -> tuple[bool, str | None]:
    """Classify a partition on WSL.

    WSL internal filesystems (root, ``/usr``, etc.) are treated as built-in.
    Windows ``drvfs`` mounts (``/mnt/c``) are mapped to physical disk bus types.

    Args:
        part: A ``psutil`` partition named tuple.

    Returns:
        ``(builtin, bus_type)`` classification.
    """
    if not _looks_like_windows_drive(part):
        return True, "WSL"

    letter = _extract_drive_letter(part)
    if not letter:
        return False, None

    bus_type = _windows_bus_type_for_letter(letter)
    return _bus_type_to_builtin(bus_type)


def _classify_linux(part: psutil._psplatform.Partition) -> tuple[bool, str | None]:
    """Classify a partition on native Linux.

    Args:
        part: A ``psutil`` partition named tuple.

    Returns:
        ``(builtin, bus_type)`` classification.
    """
    device = part.device
    if not device.startswith("/dev/"):
        return False, None

    name = device[5:]
    base = _base_block_device(name)
    if any(base.startswith(prefix) for prefix in ("loop", "ram", "zram")):
        return False, None

    removable_path = Path(f"/sys/class/block/{base}/removable")
    try:
        removable = int(removable_path.read_text(encoding="utf-8").strip())
    except (OSError, ValueError):
        removable = 0
    if removable:
        return False, "Removable"

    transport = _lsblk_transport(base)
    if transport:
        transport_upper = transport.upper()
        if transport_upper in _EXTERNAL_BUS_TYPES:
            return False, transport_upper
        return True, transport_upper

    return True, None


def _classify_windows(part: psutil._psplatform.Partition) -> tuple[bool, str | None]:
    """Classify a partition on native Windows.

    Args:
        part: A ``psutil`` partition named tuple.

    Returns:
        ``(builtin, bus_type)`` classification.
    """
    letter = _extract_drive_letter(part)
    if not letter:
        return False, None

    bus_type = _windows_bus_type_for_letter(letter)
    return _bus_type_to_builtin(bus_type)


def _looks_like_windows_drive(part: psutil._psplatform.Partition) -> bool:
    """Return True if the partition looks like a WSL drvfs Windows drive.

    Args:
        part: A ``psutil`` partition named tuple.

    Returns:
        Whether the mountpoint or device resembles a Windows drive letter.
    """
    if part.fstype and part.fstype.lower() == "9p":
        return True
    if re.match(r"^[A-Za-z]:\\?", part.device or ""):
        return True
    return bool(re.match(r"^/mnt/[A-Za-z]$", part.mountpoint or ""))


def _extract_drive_letter(part: psutil._psplatform.Partition) -> str | None:
    """Extract a single drive letter from a partition.

    Args:
        part: A ``psutil`` partition named tuple.

    Returns:
        Upper-case drive letter (e.g. ``"C"``) or ``None``.
    """
    match = re.match(r"^([A-Za-z]):\\?", part.device or "")
    if match:
        return match.group(1).upper()
    match = re.match(r"^/mnt/([A-Za-z])$", part.mountpoint or "")
    if match:
        return match.group(1).upper()
    return None


def _base_block_device(name: str) -> str:
    """Return the base block device for a partition name.

    Examples:
        ``sda1`` → ``sda``
        ``nvme0n1p1`` → ``nvme0n1``
        ``mmcblk0p1`` → ``mmcblk0``

    Args:
        name: Partition name without the ``/dev/`` prefix.

    Returns:
        Base block device name.
    """
    match = re.match(r"^(nvme\d+n\d+|mmcblk\d+)", name)
    if match:
        return match.group(1)
    match = re.match(r"^([a-z]+)", name)
    if match:
        return match.group(1)
    return name


def _lsblk_transport(name: str) -> str | None:
    """Return the transport type for a block device from ``lsblk``.

    Results are cached per base device for ``_LSBLK_CACHE_TTL_SECONDS`` to avoid
    spawning a subprocess for every partition on the same disk.

    Args:
        name: Base block device name without ``/dev/``.

    Returns:
        Lower-case transport string (e.g. ``"sata"``, ``"usb"``) or ``None``.
    """
    now = time.monotonic()
    cached = _lsblk_transport_cache.get(name)
    if cached is not None:
        transport, cached_at = cached
        if now - cached_at < _LSBLK_CACHE_TTL_SECONDS:
            return transport

    try:
        result = subprocess.run(
            ["lsblk", "-no", "TRAN", f"/dev/{name}"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="ignore",
            timeout=2,
            check=False,
        )
        transport = result.stdout.strip().lower() or None
    except (OSError, subprocess.TimeoutExpired):
        logger.debug("lsblk unavailable or failed for /dev/%s", name)
        transport = None

    _lsblk_transport_cache[name] = (transport, now)
    return transport


def _windows_bus_type_for_letter(letter: str) -> str | None:
    """Return the physical disk bus type for a Windows drive letter.

    Uses PowerShell ``Get-PhysicalDisk`` / ``Get-Partition``. Results are cached
    for ``_POWERSHELL_CACHE_TTL_SECONDS``.

    Args:
        letter: Upper-case Windows drive letter (e.g. ``"C"``).

    Returns:
        Bus type string (e.g. ``"NVMe"``, ``"USB"``) or ``None``.
    """
    mapping = _cached_windows_bus_mapping()
    return mapping.get(letter)


def _cached_windows_bus_mapping() -> dict[str, str]:
    """Return a cached drive-letter to bus-type mapping.

    Returns:
        Dictionary mapping upper-case drive letters to bus type strings.
    """
    now = time.monotonic()
    cached = _powershell_cache.get("mapping")
    if cached and now - cached[1] < _POWERSHELL_CACHE_TTL_SECONDS:
        return cached[0]

    mapping = _query_windows_bus_mapping()
    _powershell_cache["mapping"] = (mapping, now)
    return mapping


def _query_windows_bus_mapping() -> dict[str, str]:
    """Query Windows for drive-letter to bus-type mappings.

    Returns:
        Dictionary mapping upper-case drive letters to bus type strings.
    """
    script = """
    $result = Get-Partition | Where-Object { $_.DriveLetter } | ForEach-Object {
        $partition = $_
        $disk = Get-Disk -Number $partition.DiskNumber -ErrorAction SilentlyContinue
        $physical = if ($disk) {
            Get-PhysicalDisk |
                Where-Object { $_.DeviceId -eq $disk.Number } |
                Select-Object -First 1
        } else {
            Get-PhysicalDisk |
                Where-Object { $_.DeviceId -eq $partition.DiskNumber } |
                Select-Object -First 1
        }
        [PSCustomObject]@{
            DriveLetter = $partition.DriveLetter
            BusType = $physical.BusType
        }
    }
    ConvertTo-Json -InputObject @($result) -Compress
    """
    try:
        result = subprocess.run(
            ["powershell.exe", "-NoProfile", "-Command", script],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="ignore",
            timeout=10,
            check=False,
        )
        if result.returncode != 0:
            logger.debug("PowerShell bus-type query failed: %s", result.stderr)
            return {}
        data = json.loads(result.stdout or "[]")
        if isinstance(data, dict):
            data = [data]
        mapping: dict[str, str] = {}
        for item in data:
            letter = item.get("DriveLetter")
            bus = item.get("BusType")
            if letter and bus:
                mapping[str(letter).upper()] = str(bus)
        return mapping
    except (OSError, subprocess.TimeoutExpired, json.JSONDecodeError) as exc:
        logger.debug("Could not query Windows bus types: %s", exc)
        return {}


def _bus_type_to_builtin(bus_type: str | None) -> tuple[bool, str | None]:
    """Convert a bus type string into a built-in/external classification.

    Args:
        bus_type: Raw bus type string or ``None``.

    Returns:
        ``(builtin, normalised_bus_type)``. Unknown bus types are treated as
        external so that removable drives are not silently misclassified.
    """
    if not bus_type:
        return False, "Unknown"
    normalised = bus_type.upper()
    if normalised in _BUILTIN_BUS_TYPES:
        return True, normalised
    if normalised in _EXTERNAL_BUS_TYPES:
        return False, normalised
    return False, normalised
