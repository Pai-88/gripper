"""Controller slew/clamp + near_home (mode-switch safety gate)."""

from __future__ import annotations

import os
import sys
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gripper.control.controller import Controller  # noqa: E402
from gripper.messages import JointTargets  # noqa: E402
from tests._util import assert_approx  # noqa: E402


def _joints():
    def jc(mn, mx, home, rate):
        return SimpleNamespace(min_deg=mn, max_deg=mx, home_deg=home,
                               max_rate_dps=rate, invert=False, span_deg=mx - mn)
    return {"base": jc(-90, 90, 0, 120), "shoulder": jc(0, 90, 20, 90),
            "elbow": jc(-120, 0, -60, 120), "wrist": jc(-90, 90, 0, 150),
            "grip": jc(0, 100, 100, 300)}


def test_slew_caps_step():
    c = Controller(_joints(), dt=1.0 / 50)            # 20 ms tick
    cmd = c.step(JointTargets(base=90, shoulder=45, elbow=-30, wrist=10, grip=0.0))
    assert_approx(cmd.base, 120 * 0.02)               # 120 dps * 0.02 s = 2.4 deg


def test_soft_limit_clamp_flag():
    c = Controller(_joints(), dt=1.0)
    c.step(JointTargets(base=999, shoulder=20, elbow=-60, wrist=0, grip=1.0))
    assert c.clamped is True


def test_near_home_true_at_start():
    c = Controller(_joints(), dt=1.0 / 50)
    assert c.near_home() is True                      # constructed at home


def test_near_home_false_after_move():
    c = Controller(_joints(), dt=1.0)                 # 1 s tick -> big step
    c.step(JointTargets(base=80, shoulder=80, elbow=-10, wrist=40, grip=0.0))
    assert c.near_home() is False


def test_home_moves_back_toward_home():
    c = Controller(_joints(), dt=1.0)
    c.step(JointTargets(base=80, shoulder=80, elbow=-10, wrist=40, grip=0.0))
    for _ in range(200):
        c.home()
    assert c.near_home() is True
