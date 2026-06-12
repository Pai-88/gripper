"""normalize_grasp + GG-CNN map decode -> servo-error schema. The v1/v2 schema
tests are pure; the v3 decode tests need numpy and skip cleanly without it."""

from __future__ import annotations

import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gripper.vision.grasp_vision import (  # noqa: E402
    GGCNNConfig, _center_crop_box, decode_ggcnn, normalize_grasp,
)
from tests._util import assert_approx  # noqa: E402

try:
    import numpy as np
except ModuleNotFoundError:  # keep the suite zero-dep on a bare Pi
    np = None

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


# --- v3 GG-CNN decode (numpy) -------------------------------------------------

def test_center_crop_box_landscape():
    assert _center_crop_box(640, 480, 300) == (80, 0, 480)


def test_decode_ggcnn_peak_maps_to_frame():
    if np is None:
        return
    cfg = GGCNNConfig(gaussian_sigma=0.0)
    out = 8
    q = np.zeros((out, out), np.float32)
    q[2, 6] = 1.0                                  # peak at row 2, col 6
    theta = 30.0
    cos2 = np.full((out, out), math.cos(math.radians(2 * theta)), np.float32)
    sin2 = np.full((out, out), math.sin(math.radians(2 * theta)), np.float32)
    width = np.full((out, out), 0.2, np.float32)
    g = decode_ggcnn(q, cos2, sin2, width, W, H, cfg)
    # crop 480, x0=80, y0=0, scale=60 -> cx=470, cy=150
    assert_approx(g["u_norm"], (470 - 320) / 320)
    assert_approx(g["v_norm"], (150 - 240) / 240)
    assert_approx(g["theta_deg"], 30.0, tol=1e-4)
    assert_approx(g["score"], 1.0)
    assert_approx(g["width"], (0.2 * 150 * 60) / W)


def test_decode_area_frac_grows_with_region():
    if np is None:
        return
    cfg = GGCNNConfig(gaussian_sigma=0.0)
    out = 8
    o = np.ones((out, out), np.float32)
    w = np.full((out, out), 0.1, np.float32)
    small = np.zeros((out, out), np.float32); small[4, 4] = 1.0
    big = np.zeros((out, out), np.float32); big[3:6, 3:6] = 1.0
    a_small = decode_ggcnn(small, o, o, w, W, H, cfg)["area_frac"]
    a_big = decode_ggcnn(big, o, o, w, W, H, cfg)["area_frac"]
    assert a_big > a_small          # apparent-size proxy grows with the region


def test_label_decode_round_trip():
    """A grasp painted by the training label encoder decodes back to its centre,
    angle. Smoothing makes the filled disk's centroid the unique argmax."""
    if np is None:
        return
    from tools.train_ggcnn import draw_grasp_target
    out = 96
    cx, cy, theta, width_px = 60, 40, 25.0, 30.0
    q, c, s, wmap = draw_grasp_target(out, [(cx, cy, theta, width_px)], radius=8)
    g = decode_ggcnn(q, c, s, wmap, out, out, GGCNNConfig())  # default sigma>0
    assert_approx(g["u_norm"], (cx + 0.5 - out / 2) / (out / 2), tol=0.05)
    assert_approx(g["v_norm"], (cy + 0.5 - out / 2) / (out / 2), tol=0.05)
    assert_approx(g["theta_deg"], theta, tol=1e-3)
