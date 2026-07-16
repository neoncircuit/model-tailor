"""Development server launcher for the dashboard backend."""

from __future__ import annotations

import argparse
import logging
import socket
import sys
from pathlib import Path

import uvicorn

logger = logging.getLogger(__name__)


def find_free_port(start: int = 8000, host: str = "127.0.0.1") -> int:
    """Find the first available TCP port on *host* starting at *start*.

    Args:
        start: First port to try.
        host: Interface to bind to.

    Returns:
        An available port number.

    Raises:
        RuntimeError: If no free port is found before exhausting the range.
    """
    for port in range(start, 65535):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            try:
                sock.bind((host, port))
                return port
            except OSError:
                continue
    raise RuntimeError(f"No free port found starting from {start}")


def main(argv: list[str] | None = None) -> int:
    """Launch the FastAPI backend on an available port.

    Args:
        argv: Command-line arguments. If ``None``, ``sys.argv`` is used.

    Returns:
        Exit code for the process.
    """
    parser = argparse.ArgumentParser(description="Run the model-tailor dashboard backend.")
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Preferred port (default: 8000). Delegates to the next free port if taken.",
    )
    parser.add_argument(
        "--host",
        default="0.0.0.0",
        help="Host interface to bind to (default: 0.0.0.0).",
    )
    args = parser.parse_args(argv)

    script_dir = Path(__file__).resolve().parent
    src_dir = script_dir.parent / "src"
    sys.path.insert(0, str(src_dir))

    port = find_free_port(args.port, args.host)

    # Publish the chosen port so the frontend dev launcher can proxy to it.
    (script_dir.parent / ".dev-port").write_text(str(port), encoding="utf-8")

    print(f"Dashboard backend starting at http://localhost:{port}")
    logger.info("Starting uvicorn on %s:%s", args.host, port)

    uvicorn.run(
        "dashboard_api.main:create_app",
        factory=True,
        host=args.host,
        port=port,
        log_level="info",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
