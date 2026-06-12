"""Closed-form kinematics for the 4-DOF arm: base yaw + a 2-link planar
shoulder/elbow + wrist pitch. Pure ``math`` (no numpy) so it is trivially
testable and runs on the Pi with zero deps.

Frame convention
    * world z is up; base rotates about z.
    * In the arm plane, ``shoulder`` and ``elbow`` are measured such that the
      wrist position is::

          rp = L1*cos(sh) + L2*cos(sh + el)
          z  = L1*sin(sh) + L2*sin(sh + el)

      where ``rp = hypot(x, y)`` is the planar reach.
    * ``inverse`` returns the **elbow-up** branch (el <= 0).
    * ``wrist`` sets the end-effector pitch: ``pitch = sh + el + wrist``.

This intentionally models only what a 4-DOF arm can reach — no 6-DOF approach,
no dynamics, no force control (see the build guide's "honest limit for the viva").
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .messages import JointTargets


class Unreachable(ValueError):
    """Raised when a Cartesian target lies outside the arm's annulus."""


@dataclass
class ArmGeometry:
    l1: float  # shoulder -> elbow link length
    l2: float  # elbow -> wrist link length

    @property
    def reach_max(self) -> float:
        return self.l1 + self.l2

    @property
    def reach_min(self) -> float:
        return abs(self.l1 - self.l2)


def forward_position(geom: ArmGeometry, base: float, shoulder: float, elbow: float
                     ) -> tuple[float, float, float]:
    """FK -> wrist (x, y, z). Angles in radians."""
    rp = geom.l1 * math.cos(shoulder) + geom.l2 * math.cos(shoulder + elbow)
    z = geom.l1 * math.sin(shoulder) + geom.l2 * math.sin(shoulder + elbow)
    return (rp * math.cos(base), rp * math.sin(base), z)


def inverse(geom: ArmGeometry, x: float, y: float, z: float,
            approach_pitch: float = 0.0) -> JointTargets:
    """IK -> elbow-up joint angles (radians) reaching wrist position (x, y, z),
    with the end effector held at ``approach_pitch`` from horizontal.

    Raises :class:`Unreachable` if the target is outside the reachable annulus.
    """
    base = math.atan2(y, x)
    rp = math.hypot(x, y)
    d2 = rp * rp + z * z
    d = math.sqrt(d2)
    if d > geom.reach_max + 1e-9 or d < geom.reach_min - 1e-9:
        raise Unreachable(f"target reach {d:.4f} outside "
                          f"[{geom.reach_min:.4f}, {geom.reach_max:.4f}]")

    cos_el = (d2 - geom.l1 ** 2 - geom.l2 ** 2) / (2 * geom.l1 * geom.l2)
    cos_el = max(-1.0, min(1.0, cos_el))
    elbow = -math.acos(cos_el)  # elbow-up branch (negative)

    shoulder = math.atan2(z, rp) - math.atan2(
        geom.l2 * math.sin(elbow), geom.l1 + geom.l2 * math.cos(elbow)
    )
    wrist = approach_pitch - (shoulder + elbow)
    return JointTargets(base=base, shoulder=shoulder, elbow=elbow, wrist=wrist)


def end_effector_pitch(shoulder: float, elbow: float, wrist: float) -> float:
    """The world-frame pitch of the gripper given the three planar joints."""
    return shoulder + elbow + wrist


def is_reachable(geom: ArmGeometry, x: float, y: float, z: float) -> bool:
    d = math.sqrt(x * x + y * y + z * z)
    return geom.reach_min - 1e-9 <= d <= geom.reach_max + 1e-9
