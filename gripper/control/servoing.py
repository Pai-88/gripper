"""Image-based visual servoing (IBVS) + the autonomous grasp sequence — Phase 7.

The wrist camera gives an image-space grasp observation; this drives the 4-DOF
arm so the *image error* goes to zero rather than reconstructing 3D. This is what
actually delivers repeatable picks on hobby servos: it closes the residual
mechanical error (backlash + pot deadband) in image space (build guide §5).

Axis mapping (image error -> joint):
  * base  ← horizontal error  u_norm        (centre the target in x)
  * shoulder ← apparent-size error           (approach-by-scale depth proxy)
  * wrist ← grasp angle theta                (jaws perpendicular to the object)
  * grip  ← close at COMMIT

The grasp *sequence* is a small state machine: SERVO until the errors sit inside
their deadbands for ``commit_frames`` frames, then a short open-loop PLUNGE (the
trick that hides monocular depth error), CLOSE, LIFT, DONE. All pure — unit
tested in tests/test_servoing.py.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional

from ..messages import JointTargets


def wrap_grasp_angle(theta_deg: float) -> float:
    """Fold a grasp angle into (-90, 90] — a parallel gripper is symmetric."""
    while theta_deg > 90.0:
        theta_deg -= 180.0
    while theta_deg <= -90.0:
        theta_deg += 180.0
    return theta_deg


@dataclass
class GraspParams:
    # proportional gains (per-axis); signs assume the hardware conventions in §5
    kx: float = 8.0            # deg base per unit normalized-x error
    k_scale: float = 60.0      # deg shoulder per unit area-fraction error
    k_theta: float = 0.5       # deg wrist per deg grasp-angle error
    # deadbands (when inside all three for commit_frames -> COMMIT)
    x_deadband: float = 0.05
    scale_deadband: float = 0.02
    theta_deadband_deg: float = 5.0
    target_area_frac: float = 0.18  # object's apparent size at grasp distance
    commit_frames: int = 5
    # open-loop grasp moves (ticks at the control rate)
    plunge_deg: float = 12.0
    plunge_ticks: int = 10       # plunge duration cap (always enforced)
    plunge_fire_mm: Optional[float] = None  # if set + obs has range_mm (VL53L5CX),
    #                            end the plunge early when the centre is this close
    close_ticks: int = 8
    lift_deg: float = 20.0
    lift_ticks: int = 12
    max_step_deg: float = 4.0  # cap a single servo correction (the controller also slews)


class GraspPhase(Enum):
    SERVO = "SERVO"
    PLUNGE = "PLUNGE"
    CLOSE = "CLOSE"
    LIFT = "LIFT"
    DONE = "DONE"


def _clip(v: float, lim: float) -> float:
    return lim if v > lim else (-lim if v < -lim else v)


def servo_step(obs: dict, current: JointTargets, p: GraspParams
               ) -> tuple[JointTargets, bool]:
    """One IBVS correction. Returns (new target, in_tolerance)."""
    ex = obs["u_norm"]
    e_scale = p.target_area_frac - obs["area_frac"]
    e_theta = wrap_grasp_angle(obs["theta_deg"])

    d_base = -_clip(p.kx * ex, p.max_step_deg) if abs(ex) > p.x_deadband else 0.0
    d_shoulder = _clip(p.k_scale * e_scale, p.max_step_deg) \
        if abs(e_scale) > p.scale_deadband else 0.0
    d_wrist = _clip(p.k_theta * e_theta, p.max_step_deg) \
        if abs(e_theta) > p.theta_deadband_deg else 0.0

    target = JointTargets(
        base=current.base + d_base,
        shoulder=current.shoulder + d_shoulder,
        elbow=current.elbow,
        wrist=current.wrist + d_wrist,
        grip=current.grip,
    )
    in_tol = (abs(ex) <= p.x_deadband and abs(e_scale) <= p.scale_deadband
              and abs(e_theta) <= p.theta_deadband_deg)
    return target, in_tol


class GraspSequencer:
    """Drives one supervised autonomous grasp from a stream of observations."""

    def __init__(self, params: Optional[GraspParams] = None):
        self.p = params or GraspParams()
        self.phase = GraspPhase.SERVO
        self._in_tol = 0
        self._counter = 0
        self._anchor = 0.0  # shoulder angle captured at phase entry

    def done(self) -> bool:
        return self.phase is GraspPhase.DONE

    def update(self, obs: Optional[dict], current: JointTargets) -> JointTargets:
        """Advance one tick. ``obs`` may be None (lost detection during SERVO)."""
        p = self.p
        if self.phase is GraspPhase.SERVO:
            if obs is None:
                self._in_tol = 0
                return current
            target, in_tol = servo_step(obs, current, p)
            self._in_tol = self._in_tol + 1 if in_tol else 0
            if self._in_tol >= p.commit_frames:
                self.phase = GraspPhase.PLUNGE
                self._counter = 0
                self._anchor = current.shoulder
            return target

        if self.phase is GraspPhase.PLUNGE:
            self._counter += 1
            frac = self._counter / p.plunge_ticks
            target = JointTargets(**{**current.__dict__,
                                     "shoulder": self._anchor + p.plunge_deg * frac})
            # ToF gate (if configured) ends the plunge precisely; the tick count
            # is always the hard cap so a missing/blind ToF can't stall the plunge.
            tof_fired = (p.plunge_fire_mm is not None and obs is not None
                         and obs.get("range_mm") is not None
                         and obs["range_mm"] <= p.plunge_fire_mm)
            if tof_fired or self._counter >= p.plunge_ticks:
                self.phase = GraspPhase.CLOSE
                self._counter = 0
            return target

        if self.phase is GraspPhase.CLOSE:
            self._counter += 1
            target = JointTargets(**{**current.__dict__, "grip": 0.0})  # jaws closed
            if self._counter >= p.close_ticks:
                self.phase = GraspPhase.LIFT
                self._counter = 0
                self._anchor = current.shoulder
            return target

        if self.phase is GraspPhase.LIFT:
            self._counter += 1
            frac = self._counter / p.lift_ticks
            target = JointTargets(**{**current.__dict__,
                                     "shoulder": self._anchor - p.lift_deg * frac,
                                     "grip": 0.0})
            if self._counter >= p.lift_ticks:
                self.phase = GraspPhase.DONE
            return target

        return current  # DONE: hold
