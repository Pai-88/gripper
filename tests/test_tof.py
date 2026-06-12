"""VL53L5CX pure range maths — plunge gating, no hardware. The upsample test
needs numpy and skips cleanly without it."""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gripper.vision.tof import (  # noqa: E402
    _median, center_distance_mm, plunge_ready, upsample,
)
from tests._util import assert_approx  # noqa: E402

try:
    import numpy as np
except ModuleNotFoundError:
    np = None


def _grid(val: float):
    return [[val] * 8 for _ in range(8)]


def test_median():
    assert _median([3, 1, 2]) == 2
    assert_approx(_median([1, 2, 3, 4]), 2.5)
    assert _median([]) is None


def test_center_uses_central_zones():
    g = _grid(0.0)
    for r in (3, 4):
        for c in (3, 4):
            g[r][c] = 100.0
    assert_approx(center_distance_mm(g, k=2), 100.0)


def test_center_ignores_invalid_and_edges():
    g = _grid(500.0)          # the 500s sit outside the central 2x2
    g[3][3] = 0.0             # invalid -> ignored
    g[3][4] = -1.0            # invalid -> ignored
    g[4][3] = 200.0
    g[4][4] = 200.0
    assert_approx(center_distance_mm(g, k=2), 200.0)


def test_center_all_invalid_is_none():
    assert center_distance_mm(_grid(0.0)) is None


def test_plunge_ready_threshold():
    g = _grid(50.0)
    assert plunge_ready(g, fire_mm=60.0) is True
    assert plunge_ready(g, fire_mm=40.0) is False


def test_plunge_ready_false_when_blind():
    assert plunge_ready(_grid(0.0), fire_mm=1000.0) is False


def test_upsample_shape_and_corners():
    if np is None:
        return
    g = [[r * 8 + c for c in range(8)] for r in range(8)]
    up = upsample(g, 16, 24)
    assert up.shape == (24, 16)
    assert up[0, 0] == 0
    assert up[-1, -1] == g[7][7]
