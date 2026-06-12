#!/usr/bin/env python3
"""Convert Cornell / Jacquard grasp datasets into cached GG-CNN training shards.

Each raw image is centre-cropped + resized exactly the way the live detector
preprocesses a frame (gripper/vision/grasp_vision.py ``_center_crop_box`` +
``GGCNNGraspDetector._preprocess``), and its labelled grasps are rendered into
the four target maps via ``draw_grasp_target``. One ``.npz`` per sample holds
``input`` (CxHxW float32) and ``target`` (4xHxW float32); a ``manifest.txt``
lists them. ``tools/train_ggcnn.py`` then trains over that one simple format.

    python3 tools/prepare_dataset.py --dataset cornell  --data ~/raw/cornell  --out data/prep
    python3 tools/prepare_dataset.py --dataset jacquard --data ~/raw/jacquard --out data/prep --depth

The parsers (``parse_cornell_rects``, ``rect_to_grasp``, ``parse_jacquard_line``)
are pure and unit-tested; only the image I/O needs OpenCV (lazy-imported).
"""

from __future__ import annotations

import argparse
import glob
import math
import os
import pathlib
import sys
from typing import List, Optional, Sequence, Tuple

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from gripper.vision.grasp_vision import _center_crop_box  # noqa: E402
from tools.train_ggcnn import draw_grasp_target  # noqa: E402

Grasp = Tuple[float, float, float, float]  # (cx, cy, theta_deg, width_px)


# ---------------------------------------------------------------------------
# Pure parsers (unit-tested)
# ---------------------------------------------------------------------------

def parse_cornell_rects(cpos_path: str) -> List[Tuple[List[float], List[float]]]:
    """Parse a Cornell ``pcd*cpos.txt`` into a list of (xs, ys) 4-corner rects.
    Rectangles containing NaN (Cornell's invalid marker) are skipped."""
    nums = []
    with open(cpos_path) as fh:
        for line in fh:
            parts = line.split()
            if len(parts) != 2:
                continue
            try:
                nums.append((float(parts[0]), float(parts[1])))
            except ValueError:
                continue
    rects = []
    for i in range(0, len(nums) - 3, 4):
        corners = nums[i:i + 4]
        xs = [c[0] for c in corners]
        ys = [c[1] for c in corners]
        if any(math.isnan(v) for v in xs + ys):
            continue
        rects.append((xs, ys))
    return rects


def rect_to_grasp(xs: Sequence[float], ys: Sequence[float]) -> Grasp:
    """Cornell 4-corner rectangle -> (cx, cy, theta_deg, width_px).

    Convention (verify against your labels): corners ordered around the rectangle;
    the p0->p1 edge is the gripper-plate direction (grasp angle), and the p1->p2
    edge is the jaw opening (width)."""
    cx = sum(xs) / 4.0
    cy = sum(ys) / 4.0
    theta = math.degrees(math.atan2(ys[1] - ys[0], xs[1] - xs[0]))
    width = math.hypot(xs[2] - xs[1], ys[2] - ys[1])
    return cx, cy, theta, width


def parse_jacquard_line(line: str) -> Optional[Grasp]:
    """Parse one Jacquard ``*_grasps.txt`` line ``x;y;theta;opening;jaw_size``
    into (cx, cy, theta_deg, opening_px). Returns None for malformed lines."""
    parts = line.strip().split(";")
    if len(parts) < 4:
        return None
    try:
        x, y, theta, opening = (float(parts[k]) for k in range(4))
    except ValueError:
        return None
    return x, y, theta, opening


# ---------------------------------------------------------------------------
# Coordinate transform (pure) + rendering (OpenCV, lazy)
# ---------------------------------------------------------------------------

def transform_grasp(g: Grasp, x0: int, y0: int, scale: float) -> Grasp:
    """Map a raw-image grasp into the cropped+resized map space. Translation
    (crop) and uniform scaling leave the angle unchanged."""
    cx, cy, theta, width = g
    return (cx - x0) * scale, (cy - y0) * scale, theta, width * scale


def _render_input(img_bgr, depth, x0, y0, side, out_size, use_depth):
    import cv2
    import numpy as np

    if use_depth:
        crop = cv2.resize(depth[y0:y0 + side, x0:x0 + side].astype("float32"),
                          (out_size, out_size))
        crop = np.clip(crop - crop.mean(), -1.0, 1.0)   # matches detector depth path
        return crop[None].astype(np.float32)
    crop = cv2.resize(img_bgr[y0:y0 + side, x0:x0 + side], (out_size, out_size))
    rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB).astype("float32") / 255.0
    return (rgb - 0.5).transpose(2, 0, 1).astype(np.float32)


# ---------------------------------------------------------------------------
# Dataset discovery + driver
# ---------------------------------------------------------------------------

def _cornell_samples(root: str):
    for rgb in sorted(glob.glob(os.path.join(root, "**", "pcd*r.png"), recursive=True)):
        cpos = rgb.replace("r.png", "cpos.txt")
        if not os.path.exists(cpos):
            continue
        grasps = [rect_to_grasp(xs, ys) for xs, ys in parse_cornell_rects(cpos)]
        depth = rgb.replace("r.png", "d.tiff")
        yield rgb, (depth if os.path.exists(depth) else None), grasps


def _jacquard_samples(root: str):
    for rgb in sorted(glob.glob(os.path.join(root, "**", "*_RGB.png"), recursive=True)):
        gfile = rgb.replace("_RGB.png", "_grasps.txt")
        if not os.path.exists(gfile):
            continue
        grasps = []
        with open(gfile) as fh:
            for line in fh:
                g = parse_jacquard_line(line)
                if g is not None:
                    grasps.append(g)
        depth = rgb.replace("_RGB.png", "_perfect_depth.tiff")
        yield rgb, (depth if os.path.exists(depth) else None), grasps


def prepare(args) -> None:
    import cv2
    import numpy as np

    samples = (_cornell_samples if args.dataset == "cornell" else _jacquard_samples)(args.data)
    out_dir = pathlib.Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest, written, skipped = [], 0, 0
    for rgb_path, depth_path, grasps in samples:
        if args.limit and written >= args.limit:
            break
        if not grasps:
            skipped += 1
            continue
        if args.depth and depth_path is None:
            skipped += 1
            continue
        img = cv2.imread(rgb_path)
        if img is None:
            skipped += 1
            continue
        h, w = img.shape[:2]
        x0, y0, side = _center_crop_box(w, h, args.out_size)
        scale = args.out_size / side
        mapped = [transform_grasp(g, x0, y0, scale) for g in grasps]
        depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED) if args.depth else None

        inp = _render_input(img, depth, x0, y0, side, args.out_size, args.depth)
        q, c, s, wmap = draw_grasp_target(args.out_size, mapped)
        target = np.stack([q, c, s, wmap], 0).astype(np.float32)

        name = f"sample_{written:06d}.npz"
        np.savez_compressed(out_dir / name, input=inp, target=target)
        manifest.append(name)
        written += 1
        if written % 100 == 0:
            print(f"  {written} written...")

    (out_dir / "manifest.txt").write_text("\n".join(manifest) + ("\n" if manifest else ""))
    print(f"prepared {written} samples -> {out_dir}  (skipped {skipped})")
    if not written:
        raise SystemExit(
            f"no usable samples found under {args.data} for --dataset {args.dataset}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True, choices=["cornell", "jacquard"])
    ap.add_argument("--data", required=True, help="raw dataset root")
    ap.add_argument("--out", default="data/prep", help="output dir for .npz shards")
    ap.add_argument("--out-size", type=int, default=300,
                    help="square map side (must match train --out-size; div by 12)")
    ap.add_argument("--depth", action="store_true",
                    help="render the 1-channel depth input (needs depth files)")
    ap.add_argument("--limit", type=int, default=0, help="cap samples (0 = all)")
    args = ap.parse_args()
    if args.out_size % 12:
        ap.error("--out-size must be divisible by 12 (e.g. 300)")
    prepare(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
