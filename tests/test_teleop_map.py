"""One-Euro filter behaviour and the pinch->grip / palm->reach mapping."""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gripper.control.teleop_map import (  # noqa: E402
    OneEuroFilter, Calibration, landmarks_to_targets,
)
from tests._util import assert_approx  # noqa: E402


def _hand(thumb_index_gap: float, palm: float = 0.2, cx: float = 0.5):
    """Build 21 landmarks where only the indices the mapping reads are meaningful."""
    lm = [(0.0, 0.0, 0.0)] * 21
    lm[0] = (cx, 0.5, 0.0)            # wrist
    lm[9] = (cx, 0.5 - palm, 0.0)     # middle MCP -> palm size = `palm`
    lm[5] = (cx, 0.45, 0.0)           # index MCP
    lm[8] = (cx, 0.40, 0.0)           # index tip
    lm[4] = (cx + thumb_index_gap, 0.40, 0.0)  # thumb tip, offset from index tip
    return lm


def test_one_euro_passes_first_sample():
    f = OneEuroFilter(freq=30)
    assert f(5.0) == 5.0


def test_one_euro_holds_constant():
    f = OneEuroFilter(freq=30)
    for _ in range(50):
        out = f(7.0)
    assert_approx(out, 7.0, 1e-9)


def test_one_euro_attenuates_then_converges():
    f = OneEuroFilter(freq=30, mincutoff=1.0, beta=0.0)
    out = f(0.0)
    out = f(10.0)
    assert 0.0 < out < 10.0          # first step lags (smoothed)
    for _ in range(200):
        out = f(10.0)
    assert_approx(out, 10.0, 1e-3)   # eventually converges


def test_grip_monotonic_in_pinch():
    cal = Calibration()
    closed = landmarks_to_targets(_hand(thumb_index_gap=0.04), cal)
    open_ = landmarks_to_targets(_hand(thumb_index_gap=0.24), cal)
    assert closed.grip < open_.grip
    assert 0.0 <= closed.grip <= 1.0
    assert 0.0 <= open_.grip <= 1.0


def test_base_follows_hand_x():
    cal = Calibration()
    left = landmarks_to_targets(_hand(0.1, cx=0.25), cal)
    right = landmarks_to_targets(_hand(0.1, cx=0.75), cal)
    assert left.base < right.base
