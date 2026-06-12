"""Latency instrumentation + the dashboard snapshot — pure helpers so the
control loop stays thin and both are unit-testable.

"Median glass-to-servo 60 ms, p95 95 ms" is exactly the quantitative rigour an
interviewer wants (build guide, "Latency instrumentation").
"""

from __future__ import annotations

from collections import deque
from typing import Any


class LatencyTracker:
    """Rolling window of latency samples (ms) with p50/p95 readout."""

    def __init__(self, window: int = 200):
        self._d: deque[float] = deque(maxlen=window)

    def add(self, ms: float) -> None:
        self._d.append(float(ms))

    def __len__(self) -> int:
        return len(self._d)

    def percentile(self, q: float) -> float:
        if not self._d:
            return 0.0
        s = sorted(self._d)
        i = min(len(s) - 1, int(q / 100 * len(s)))
        return s[i]

    def p50(self) -> float:
        return self.percentile(50)

    def p95(self) -> float:
        return self.percentile(95)


def build_snapshot(state: str, cmd: Any, flags: int = 0, vbat_mV: int = 0,
                   p50: float = 0.0, p95: float = 0.0, clamped: bool = False,
                   seq: int = 0) -> dict:
    """Assemble the JSON-serialisable status dict streamed to the dashboard.
    ``cmd`` is a JointTargets-like object (duck-typed)."""
    return {
        "state": state,
        "joints": {n: round(getattr(cmd, n), 1)
                   for n in ("base", "shoulder", "elbow", "wrist")},
        "grip": round(cmd.grip, 3),
        "flags": int(flags),
        "vbat_mV": int(vbat_mV),
        "latency_ms": {"p50": round(p50, 1), "p95": round(p95, 1)},
        "clamped": bool(clamped),
        "seq": int(seq),
    }
