"""Clamp / rate-limit / slew correctness."""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gripper.control import limits as L  # noqa: E402
from tests._util import assert_approx, assert_raises  # noqa: E402


def test_clamp_basic():
    assert L.clamp(5, 0, 10) == 5
    assert L.clamp(-1, 0, 10) == 0
    assert L.clamp(11, 0, 10) == 10


def test_clamp_swapped_bounds():
    assert L.clamp(5, 10, 0) == 5
    assert L.clamp(-3, 10, 0) == 0


def test_rate_limit_caps_step():
    assert L.rate_limit(100, 0, 5) == 5      # moving up, capped
    assert L.rate_limit(-100, 0, 5) == -5    # moving down, capped
    assert L.rate_limit(3, 0, 5) == 3        # within step, reaches target


def test_rate_limit_negative_delta_rejected():
    assert_raises(ValueError, L.rate_limit, 1, 0, -1)


def test_slew_uses_rate_and_dt():
    # 120 deg/s for 10 ms -> max 1.2 deg.
    assert_approx(L.slew(100, 0, 120, 0.01), 1.2)


def test_clamp_and_slew_flags_clamp():
    value, hit = L.clamp_and_slew(target=200, current=0, lo=0, hi=90,
                                  max_rate_dps=120, dt=1.0)
    assert hit is True              # 200 exceeded the 90 soft limit
    assert value == 90              # slew step (120) exceeds remaining gap to 90

    value2, hit2 = L.clamp_and_slew(target=45, current=0, lo=0, hi=90,
                                    max_rate_dps=10, dt=1.0)
    assert hit2 is False
    assert value2 == 10             # rate-limited toward an in-range target
