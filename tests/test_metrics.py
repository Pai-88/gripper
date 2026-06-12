"""LatencyTracker percentiles + build_snapshot shape."""

from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gripper.metrics import LatencyTracker, build_snapshot  # noqa: E402
from gripper.messages import JointTargets  # noqa: E402


def test_empty_tracker_is_zero():
    t = LatencyTracker()
    assert t.p50() == 0.0 and t.p95() == 0.0
    assert len(t) == 0


def test_percentiles():
    t = LatencyTracker(window=1000)
    for v in range(1, 101):  # 1..100
        t.add(v)
    assert t.p50() == 51
    assert t.p95() == 96
    assert len(t) == 100


def test_window_evicts_old_samples():
    t = LatencyTracker(window=5)
    for v in range(100):
        t.add(v)
    assert len(t) == 5
    assert t.p50() >= 95  # only the most recent survive


def test_build_snapshot_shape_and_json():
    cmd = JointTargets(base=12.0, shoulder=20.04, elbow=-60.1, wrist=0.0, grip=0.512)
    snap = build_snapshot("TELEOP", cmd, flags=2, vbat_mV=7380,
                          p50=58.4, p95=94.9, clamped=True, seq=42)
    assert snap["state"] == "TELEOP"
    assert snap["joints"] == {"base": 12.0, "shoulder": 20.0, "elbow": -60.1, "wrist": 0.0}
    assert snap["grip"] == 0.512
    assert snap["latency_ms"] == {"p50": 58.4, "p95": 94.9}
    assert snap["clamped"] is True
    assert snap["vbat_mV"] == 7380 and snap["seq"] == 42
    json.dumps(snap)  # must be serialisable for the websocket
