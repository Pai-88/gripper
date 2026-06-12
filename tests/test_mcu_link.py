"""targets_to_wire: correctness, out-of-range clamping, and non-finite safety
(regression for the stress-test finding — NaN/inf must never crash the wire)."""

from __future__ import annotations

import math
import os
import sys
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gripper.comms.mcu_link import targets_to_wire  # noqa: E402
from gripper.messages import JointTargets  # noqa: E402
from gripper import protocol as p  # noqa: E402


def _joints():
    def jc(mn, mx, invert=False):
        return SimpleNamespace(min_deg=mn, max_deg=mx, invert=invert, span_deg=mx - mn)
    return {"base": jc(-90, 90), "shoulder": jc(0, 90), "elbow": jc(-120, 0),
            "wrist": jc(-90, 90), "grip": jc(0, 100)}


def test_offset_from_min_encoding():
    cd, grip = targets_to_wire(
        JointTargets(base=0, shoulder=20, elbow=-60, wrist=0, grip=0.5), _joints())
    assert cd == [9000, 2000, 6000, 9000]  # (angle - min) * 100
    assert grip == 500


def test_invert_mirrors():
    j = _joints()
    j["base"] = SimpleNamespace(min_deg=-90, max_deg=90, invert=True, span_deg=180)
    cd, _ = targets_to_wire(JointTargets(base=0, shoulder=20, elbow=-60, wrist=0, grip=1), j)
    assert cd[0] == 9000  # mirror of centre stays centre


def test_out_of_range_clamps_into_wire_bounds():
    cd, grip = targets_to_wire(
        JointTargets(base=1e6, shoulder=-1e6, elbow=0, wrist=0, grip=9.0), _joints())
    assert all(p.ANGLE_MIN_CD <= a <= p.ANGLE_MAX_CD for a in cd)
    assert grip == p.GRIP_MAX


def test_non_finite_is_failsafe_not_a_crash():
    for bad in (math.nan, math.inf, -math.inf):
        cd, grip = targets_to_wire(
            JointTargets(base=bad, shoulder=bad, elbow=bad, wrist=bad, grip=bad), _joints())
        assert all(isinstance(a, int) and p.ANGLE_MIN_CD <= a <= p.ANGLE_MAX_CD for a in cd)
        assert cd == [0, 0, 0, 0]      # parked at each joint's min
        assert grip == 0               # grip fails safe to closed
