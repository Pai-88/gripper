"""Mode 1 mapping: 21 MediaPipe hand landmarks -> joint targets, with a
per-joint One-Euro filter for jitter-free, low-lag teleoperation.

Pure math on ``(x, y, z)`` landmark tuples — no MediaPipe import — so the
mapping and the filter are unit-testable. Landmark indices follow MediaPipe:
0 wrist, 4 thumb-tip, 5 index-MCP, 8 index-tip, 9 middle-MCP.

Every grip-relevant distance is normalized by palm size ``s = dist(0, 9)`` so
the pinch threshold does not drift as the hand moves toward/away from the camera.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

from ..messages import JointTargets

Landmark = tuple  # (x, y, z), x/y normalized to [0, 1]


def _dist(a: Landmark, b: Landmark) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _map_range(v: float, lo: float, hi: float, out_lo: float, out_hi: float) -> float:
    if hi == lo:
        return out_lo
    t = (v - lo) / (hi - lo)
    t = max(0.0, min(1.0, t))
    return out_lo + t * (out_hi - out_lo)


def _signed_angle(u: tuple, v: tuple) -> float:
    """Signed angle (radians) from 2D vector u to v."""
    return math.atan2(u[0] * v[1] - u[1] * v[0], u[0] * v[0] + u[1] * v[1])


@dataclass
class Calibration:
    """Per-user ranges captured by ``tools/calibrate.py`` (normalized image space)."""

    x_left: float = 0.20
    x_right: float = 0.80
    y_top: float = 0.20
    y_bot: float = 0.80
    s_near: float = 0.35   # palm size when hand is close
    s_far: float = 0.12    # palm size when hand is far
    flex_min: float = -0.6
    flex_max: float = 0.6
    pinch_closed: float = 0.25  # thumb-index dist / palm size
    pinch_open: float = 1.20
    base_span_deg: float = 90.0
    sh_min_deg: float = 10.0
    sh_max_deg: float = 80.0
    wr_min_deg: float = -60.0
    wr_max_deg: float = 60.0


def landmarks_to_targets(lm: Sequence[Landmark], cal: Calibration) -> JointTargets:
    wrist, mid_mcp = lm[0], lm[9]
    idx_tip, thb_tip, idx_mcp = lm[8], lm[4], lm[5]

    s = _dist(wrist, mid_mcp) or 1e-6  # palm scale / depth proxy

    # BASE: horizontal palm-centre position (less noisy than forearm angle).
    cx = 0.5 * (wrist[0] + mid_mcp[0])
    base = _map_range(cx, cal.x_left, cal.x_right, -1.0, 1.0) * cal.base_span_deg

    # SHOULDER/REACH: palm size (depth) blended with vertical position.
    reach = _map_range(s, cal.s_far, cal.s_near, 0.0, 1.0)
    cy = 0.5 * (wrist[1] + mid_mcp[1])
    lift = _map_range(cy, cal.y_top, cal.y_bot, 1.0, 0.0)
    shoulder = _map_range(0.5 * reach + 0.5 * lift, 0.0, 1.0,
                          cal.sh_min_deg, cal.sh_max_deg)

    # WRIST FLEX: signed angle of (idx_mcp -> idx_tip) vs the hand axis.
    hand_axis = (mid_mcp[0] - wrist[0], mid_mcp[1] - wrist[1])
    finger = (idx_tip[0] - idx_mcp[0], idx_tip[1] - idx_mcp[1])
    flex = _signed_angle(hand_axis, finger)
    wrist_deg = _map_range(flex, cal.flex_min, cal.flex_max,
                           cal.wr_min_deg, cal.wr_max_deg)

    # GRIPPER: thumb-index pinch, normalized by palm size (scale-invariant).
    pinch = _dist(thb_tip, idx_tip) / s
    grip = _map_range(pinch, cal.pinch_closed, cal.pinch_open, 0.0, 1.0)

    # Teleop uses direct per-joint mapping; the elbow is coupled to the
    # shoulder here (auto mode solves it via IK instead).
    elbow = -shoulder * 0.5
    return JointTargets(base=base, shoulder=shoulder, elbow=elbow,
                        wrist=wrist_deg, grip=grip)


class OneEuroFilter:
    """Adaptive low-pass filter: heavy smoothing at rest (kills tremor), light
    at speed (low lag). Apply one per scalar joint target. Reference:
    Casiez et al., "1€ Filter" (CHI 2012)."""

    def __init__(self, freq: float, mincutoff: float = 1.0,
                 beta: float = 0.007, dcutoff: float = 1.0):
        self.freq = freq
        self.mincutoff = mincutoff
        self.beta = beta
        self.dcutoff = dcutoff
        self._x_prev: float | None = None
        self._dx_prev = 0.0

    def _alpha(self, cutoff: float) -> float:
        tau = 1.0 / (2 * math.pi * cutoff)
        te = 1.0 / self.freq
        return 1.0 / (1.0 + tau / te)

    def __call__(self, x: float) -> float:
        if self._x_prev is None:
            self._x_prev = x
            return x
        dx = (x - self._x_prev) * self.freq
        a_d = self._alpha(self.dcutoff)
        dx_hat = a_d * dx + (1 - a_d) * self._dx_prev
        cutoff = self.mincutoff + self.beta * abs(dx_hat)
        a = self._alpha(cutoff)
        x_hat = a * x + (1 - a) * self._x_prev
        self._x_prev = x_hat
        self._dx_prev = dx_hat
        return x_hat
