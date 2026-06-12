"""IK ∘ FK round-trips and reachability edges for the 2-link arm."""

from __future__ import annotations

import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gripper import kinematics as k  # noqa: E402
from tests._util import assert_approx, assert_raises  # noqa: E402

GEOM = k.ArmGeometry(l1=0.12, l2=0.10)  # metres, xArm-ish


def test_fk_ik_position_consistency():
    # For a spread of valid elbow-up poses, IK must reproduce the FK position.
    for base in (-1.0, 0.0, 0.7):
        for sh in (0.2, 0.6, 1.0):
            for el in (-1.2, -0.6, -0.2):
                x, y, z = k.forward_position(GEOM, base, sh, el)
                sol = k.inverse(GEOM, x, y, z)
                x2, y2, z2 = k.forward_position(GEOM, sol.base, sol.shoulder, sol.elbow)
                assert_approx(x, x2, 1e-6, "x")
                assert_approx(y, y2, 1e-6, "y")
                assert_approx(z, z2, 1e-6, "z")


def test_inverse_returns_elbow_up():
    sol = k.inverse(GEOM, 0.15, 0.0, 0.05)
    assert sol.elbow <= 0.0  # elbow-up branch


def test_unreachable_too_far():
    assert_raises(k.Unreachable, k.inverse, GEOM, 0.30, 0.0, 0.0)  # > l1+l2


def test_unreachable_too_close():
    # Inside the inner radius |l1 - l2| = 0.02 m.
    assert_raises(k.Unreachable, k.inverse, GEOM, 0.01, 0.0, 0.0)


def test_is_reachable_matches_inverse():
    pt = (0.15, 0.02, 0.04)
    assert k.is_reachable(GEOM, *pt)
    sol = k.inverse(GEOM, *pt)
    assert isinstance(sol.shoulder, float)


def test_end_effector_pitch_from_inverse():
    sol = k.inverse(GEOM, 0.14, 0.03, 0.05, approach_pitch=0.3)
    pitch = k.end_effector_pitch(sol.shoulder, sol.elbow, sol.wrist)
    assert_approx(pitch, 0.3, 1e-6)
