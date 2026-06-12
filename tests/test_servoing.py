"""IBVS control law + grasp sequencer (Phase 7). Closed-loop convergence is
checked with a toy negative-feedback plant — no camera/arm needed."""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gripper.control.servoing import (  # noqa: E402
    GraspParams, GraspPhase, GraspSequencer, servo_step, wrap_grasp_angle,
)
from gripper.messages import JointTargets  # noqa: E402
from tests._util import assert_approx  # noqa: E402


def _home():
    return JointTargets(base=0.0, shoulder=20.0, elbow=-60.0, wrist=0.0, grip=1.0)


def test_wrap_grasp_angle():
    assert wrap_grasp_angle(45) == 45
    assert wrap_grasp_angle(100) == -80
    assert wrap_grasp_angle(-120) == 60
    assert wrap_grasp_angle(-90) == 90  # symmetric jaw


def test_centred_target_is_in_tolerance_and_still():
    p = GraspParams()
    obs = {"u_norm": 0.0, "area_frac": p.target_area_frac, "theta_deg": 0.0}
    cur = _home()
    tgt, in_tol = servo_step(obs, cur, p)
    assert in_tol is True
    assert tgt.base == cur.base and tgt.shoulder == cur.shoulder and tgt.wrist == cur.wrist


def test_servo_corrections_have_right_sign():
    p = GraspParams()
    cur = _home()
    # target to the right (u_norm > 0) -> base decreases
    tgt, _ = servo_step({"u_norm": 0.3, "area_frac": p.target_area_frac, "theta_deg": 0}, cur, p)
    assert tgt.base < cur.base
    # object too small (far) -> shoulder advances
    tgt, _ = servo_step({"u_norm": 0, "area_frac": 0.05, "theta_deg": 0}, cur, p)
    assert tgt.shoulder > cur.shoulder


def test_single_correction_is_capped():
    p = GraspParams(max_step_deg=4.0)
    tgt, _ = servo_step({"u_norm": 1.0, "area_frac": 0.0, "theta_deg": 90}, _home(), p)
    assert abs(tgt.base - 0.0) <= 4.0 + 1e-9


def test_sequencer_phase_progression():
    p = GraspParams(commit_frames=3, plunge_ticks=2, close_ticks=2, lift_ticks=2)
    seq = GraspSequencer(p)
    cur = _home()
    centred = {"u_norm": 0.0, "area_frac": p.target_area_frac, "theta_deg": 0.0}
    for _ in range(3):
        assert seq.phase is GraspPhase.SERVO
        seq.update(centred, cur)
    assert seq.phase is GraspPhase.PLUNGE

    saw_closed = False
    for _ in range(30):
        t = seq.update(None, cur)
        if t.grip == 0.0:
            saw_closed = True
        if seq.done():
            break
    assert saw_closed, "gripper never commanded closed"
    assert seq.done()


def test_lost_detection_holds_and_resets_commit():
    p = GraspParams(commit_frames=3)
    seq = GraspSequencer(p)
    cur = _home()
    centred = {"u_norm": 0.0, "area_frac": p.target_area_frac, "theta_deg": 0.0}
    seq.update(centred, cur)
    seq.update(centred, cur)
    held = seq.update(None, cur)         # detection lost -> hold, reset counter
    assert held.base == cur.base and seq.phase is GraspPhase.SERVO
    # one in-tol frame isn't enough now; must rebuild the streak
    seq.update(centred, cur)
    seq.update(centred, cur)
    assert seq.phase is GraspPhase.SERVO
    seq.update(centred, cur)
    assert seq.phase is GraspPhase.PLUNGE


def test_tof_gated_plunge_fires_early():
    p = GraspParams(commit_frames=2, plunge_ticks=50, plunge_fire_mm=80.0)
    seq = GraspSequencer(p)
    cur = _home()
    centred = {"u_norm": 0.0, "area_frac": p.target_area_frac, "theta_deg": 0.0}
    seq.update(centred, cur)
    seq.update(centred, cur)
    assert seq.phase is GraspPhase.PLUNGE
    seq.update({"range_mm": 200.0}, cur)        # still far -> keep plunging
    assert seq.phase is GraspPhase.PLUNGE
    seq.update({"range_mm": 50.0}, cur)         # within fire_mm -> CLOSE early (tick 2 << 50)
    assert seq.phase is GraspPhase.CLOSE


def test_plunge_ignores_range_when_gate_disabled():
    p = GraspParams(commit_frames=1, plunge_ticks=5)   # plunge_fire_mm is None
    seq = GraspSequencer(p)
    cur = _home()
    seq.update({"u_norm": 0.0, "area_frac": p.target_area_frac, "theta_deg": 0.0}, cur)
    assert seq.phase is GraspPhase.PLUNGE
    seq.update({"range_mm": 1.0}, cur)          # very close, but gate off -> ticks only
    assert seq.phase is GraspPhase.PLUNGE


def test_closed_loop_converges_and_completes():
    """Toy plant: each commanded joint delta nudges the image error toward 0."""
    p = GraspParams(commit_frames=3)
    seq = GraspSequencer(p)
    cur = _home()
    u, area, theta = 0.5, 0.05, 30.0
    done = False
    for _ in range(600):
        obs = {"u_norm": u, "area_frac": area, "theta_deg": theta}
        t = seq.update(obs, cur)
        if seq.phase is GraspPhase.SERVO:           # plant responds during servoing
            u += (t.base - cur.base) * 0.06         # base move centres the target
            area += (t.shoulder - cur.shoulder) * 0.003   # advance -> looks bigger
            theta -= (t.wrist - cur.wrist) * 0.06   # wrist rotation aligns jaws
        cur = t
        if seq.done():
            done = True
            break
    assert done, "grasp never completed"
    assert abs(u) <= p.x_deadband + 1e-6
    assert abs(theta) <= p.theta_deadband_deg + 1e-6
