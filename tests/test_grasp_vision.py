"""normalize_grasp: pixel detection -> servo-error schema (pure, no opencv)."""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gripper.vision.grasp_vision import normalize_grasp  # noqa: E402
from tests._util import assert_approx  # noqa: E402

W, H = 640, 480


def test_centred_object_zero_error():
    g = normalize_grasp(W / 2, H / 2, 100, 50, 0.0, W, H)
    assert g["u_norm"] == 0.0 and g["v_norm"] == 0.0


def test_right_edge_is_plus_one():
    g = normalize_grasp(W, H / 2, 40, 40, 0.0, W, H)
    assert_approx(g["u_norm"], 1.0)


def test_top_edge_is_minus_one():
    g = normalize_grasp(W / 2, 0, 40, 40, 0.0, W, H)
    assert_approx(g["v_norm"], -1.0)


def test_area_fraction():
    g = normalize_grasp(W / 2, H / 2, 100, 50, 0.0, W, H)
    assert_approx(g["area_frac"], (100 * 50) / (W * H))


def test_angle_is_wrapped():
    g = normalize_grasp(W / 2, H / 2, 80, 30, 100.0, W, H)
    assert g["theta_deg"] == -80.0  # 100 folded into (-90, 90]


def test_width_is_short_side_fraction():
    g = normalize_grasp(W / 2, H / 2, 120, 40, 0.0, W, H)
    assert_approx(g["width"], 40 / W)


def test_schema_is_json_serialisable():
    import json
    json.dumps(normalize_grasp(10, 20, 30, 40, 12.0, W, H))
