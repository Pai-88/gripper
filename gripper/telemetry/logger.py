"""Structured JSONL logging — your reimplemented ``ros2 bag``. One record per
state transition / command / fault, grep-able and pandas-loadable. Falls back to
the stdlib ``logging`` if ``structlog`` is not installed so imports never fail.
"""

from __future__ import annotations

import json
import pathlib
import time
from typing import Any

try:
    import structlog  # type: ignore
    _HAVE_STRUCTLOG = True
except ImportError:
    _HAVE_STRUCTLOG = False


class JsonlLogger:
    """Minimal monotonic-stamped JSONL sink, used whether or not structlog exists."""

    def __init__(self, run_dir: str | pathlib.Path = "logs"):
        run_dir = pathlib.Path(run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%Y%m%d_%H%M%S")
        self.path = run_dir / f"run_{stamp}.jsonl"
        self._f = open(self.path, "a", buffering=1)
        self._t0 = time.monotonic()

    def log(self, event: str, **fields: Any) -> None:
        rec = {"t_ms": round((time.monotonic() - self._t0) * 1000, 1),
               "event": event, **fields}
        self._f.write(json.dumps(rec, default=str) + "\n")

    def close(self) -> None:
        self._f.close()


def get_logger(run_dir: str | pathlib.Path = "logs") -> JsonlLogger:
    """Return a JSONL logger. (structlog, when present, can be wired to the same
    file sink; the simple sink is intentionally dependency-free.)"""
    return JsonlLogger(run_dir)
