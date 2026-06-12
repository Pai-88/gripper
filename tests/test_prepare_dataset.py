"""Pure dataset-prep parsers + coordinate transform (no OpenCV/disk). The
render->decode round trip needs numpy and skips cleanly without it."""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.prepare_dataset import (  # noqa: E402
    parse_jacquard_line, rect_to_grasp, transform_grasp,
)
from tests._util import assert_approx  # noqa: E402

try:
    import numpy as np
except ModuleNotFoundError:
    np = None


def test_parse_jacquard_line():
    assert parse_jacquard_line("100;200;30;40;12") == (100.0, 200.0, 30.0, 40.0)
    assert parse_jacquard_line("1;2;3;4") == (1.0, 2.0, 3.0, 4.0)  # jaw_size optional
    assert parse_jacquard_line("bad;line") is None
    assert parse_jacquard_line("") is None
    assert parse_jacquard_line("a;b;c;d") is None


def test_rect_to_grasp_axis_aligned():
    # corners p0(0,0) p1(40,0) p2(40,20) p3(0,20): horizontal plate, 20px opening
    cx, cy, theta, width = rect_to_grasp([0, 40, 40, 0], [0, 0, 20, 20])
    assert_approx(cx, 20.0)
    assert_approx(cy, 10.0)
    assert_approx(theta, 0.0)
    assert_approx(width, 20.0)


def test_rect_to_grasp_angled():
    # p0->p1 at +45deg
    _, _, theta, _ = rect_to_grasp([0, 10, 5, -5], [0, 10, 15, 5])
    assert_approx(theta, 45.0)


def test_transform_grasp_crop_and_scale():
    # crop offset (80,0), scale 0.5: centre shifts + width scales, angle unchanged
    cx, cy, theta, width = transform_grasp((120.0, 40.0, 30.0, 60.0), 80, 0, 0.5)
    assert_approx(cx, 20.0)        # (120-80)*0.5
    assert_approx(cy, 20.0)        # (40-0)*0.5
    assert_approx(theta, 30.0)     # invariant under translate + uniform scale
    assert_approx(width, 30.0)     # 60*0.5


def test_prepared_target_decodes_back():
    """A grasp transformed + rendered the way prepare does decodes to its mapped
    centre/angle — closing the prepare -> train -> infer loop."""
    if np is None:
        return
    from tools.train_ggcnn import draw_grasp_target
    from gripper.vision.grasp_vision import GGCNNConfig, decode_ggcnn

    out = 96
    raw = (140.0, 90.0, 25.0, 40.0)            # raw-image grasp
    x0, y0, scale = 50, 30, out / 120.0        # pretend a 120px crop -> 96
    g = transform_grasp(raw, x0, y0, scale)
    q, c, s, wmap = draw_grasp_target(out, [g], radius=8)
    dec = decode_ggcnn(q, c, s, wmap, out, out, GGCNNConfig())
    assert_approx(dec["u_norm"], (g[0] + 0.5 - out / 2) / (out / 2), tol=0.05)
    assert_approx(dec["v_norm"], (g[1] + 0.5 - out / 2) / (out / 2), tol=0.05)
    assert_approx(dec["theta_deg"], 25.0, tol=1e-3)
