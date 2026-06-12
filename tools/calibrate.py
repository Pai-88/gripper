#!/usr/bin/env python3
"""Guided calibration -> config/calibration.yaml.

Captures (1) the per-user teleop ranges (reach/position span, wrist flex, pinch)
and (2) the eye-in-hand camera intrinsics + hand-eye transform, so the same
pixel->pose mapping holds run to run.

This is a skeleton: the prompts/printouts are wired, the capture steps are TODO
(they need MediaPipe + OpenCV at the bench). See the build guide,
"Calibration routine" and "Camera mounting".

    python3 tools/calibrate.py --teleop      # per-user teleop ranges
    python3 tools/calibrate.py --intrinsics  # cv2.calibrateCamera on a chessboard
"""

from __future__ import annotations

import argparse
import pathlib

import yaml

ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT = ROOT / "config" / "calibration.yaml"


def calibrate_teleop() -> dict:
    print("Teleop calibration — follow the prompts (hold each pose ~1 s):")
    steps = ["hand CLOSE then FAR", "hand fully LEFT/RIGHT/TOP/BOTTOM",
             "wrist FLEX up then down", "pinch CLOSED then OPEN wide"]
    for i, s in enumerate(steps, 1):
        print(f"  [{i}/{len(steps)}] {s}")
    # TODO: run MediaPipe, record min/max of palm size, x/y, flex angle, pinch.
    # Bias each captured range inward ~5-10% so extremes are comfortable.
    return {"teleop": {"note": "TODO: fill from capture",
                       "x_left": 0.2, "x_right": 0.8}}


def calibrate_intrinsics() -> dict:
    print("Intrinsics — show a chessboard to the wrist camera from several angles.")
    # TODO: cv2.findChessboardCorners + cv2.calibrateCamera; optionally
    # cv2.calibrateHandEye for the gripper-frame transform.
    return {"camera": {"note": "TODO: fill from cv2.calibrateCamera",
                       "fx": None, "fy": None, "cx": None, "cy": None}}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--teleop", action="store_true")
    ap.add_argument("--intrinsics", action="store_true")
    args = ap.parse_args()

    data = yaml.safe_load(OUT.read_text()) if OUT.exists() else {}
    if args.teleop or not (args.teleop or args.intrinsics):
        data.update(calibrate_teleop())
    if args.intrinsics:
        data.update(calibrate_intrinsics())
    OUT.write_text(yaml.safe_dump(data, sort_keys=False))
    print(f"wrote {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
