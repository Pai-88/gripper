"""The controller: turns a requested :class:`JointTargets` into a safe, smoothed
command by clamping to per-joint soft limits and rate-limiting (slew) at the
loop rate. It is the *single writer* of commanded joint state.

Re-clamped independently on the ESP32 — this is the primary, not the only, line
of defence.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .. import JOINT_NAMES
from ..messages import JointTargets
from . import limits

if TYPE_CHECKING:  # JointCfg is only used in annotations (lazy via __future__)
    from ..config import JointCfg


class Controller:
    def __init__(self, joints: dict[str, JointCfg], dt: float):
        self.joints = joints
        self.dt = dt
        # Start at each joint's home so the first command doesn't lurch.
        self._current = JointTargets(
            base=joints["base"].home_deg,
            shoulder=joints["shoulder"].home_deg,
            elbow=joints["elbow"].home_deg,
            wrist=joints["wrist"].home_deg,
            grip=joints["grip"].home_deg / 100.0,
        )
        self.clamped = False

    @property
    def current(self) -> JointTargets:
        return self._current

    def near_home(self, tol_deg: float = 5.0) -> bool:
        """True when every arm joint is within ``tol_deg`` of its home angle.
        Used to gate mode switches to safe (slow, near-home) states."""
        for name in JOINT_NAMES:
            if abs(self._current.__dict__[name] - self.joints[name].home_deg) > tol_deg:
                return False
        return True

    def home(self) -> JointTargets:
        """Snap the command toward home (still slew-limited by step())."""
        return self.step(JointTargets(
            base=self.joints["base"].home_deg,
            shoulder=self.joints["shoulder"].home_deg,
            elbow=self.joints["elbow"].home_deg,
            wrist=self.joints["wrist"].home_deg,
            grip=self.joints["grip"].home_deg / 100.0,
        ))

    def step(self, request: JointTargets) -> JointTargets:
        """Apply clamp + slew to one requested target; advance internal state."""
        hit_any = False
        new = JointTargets(grip=self._current.grip)
        for name in JOINT_NAMES:
            cfg = self.joints[name]
            value, hit = limits.clamp_and_slew(
                target=request.__dict__[name],
                current=self._current.__dict__[name],
                lo=cfg.min_deg, hi=cfg.max_deg,
                max_rate_dps=cfg.max_rate_dps, dt=self.dt,
            )
            setattr(new, name, value)
            hit_any = hit_any or hit
        # Grip uses a 0..1 fraction; rate-limit it gently too.
        gcfg = self.joints["grip"]
        grip_target = limits.clamp(request.grip, 0.0, 1.0)
        new.grip = limits.slew(grip_target, self._current.grip,
                               gcfg.max_rate_dps / 100.0, self.dt)
        self._current = new
        self.clamped = hit_any
        return new
