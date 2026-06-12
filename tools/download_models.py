#!/usr/bin/env python3
"""Fetch the ML model assets the stack needs into ./models/.

    python3 tools/download_models.py            # teleop gesture recognizer
    python3 tools/download_models.py --all

The YOLO grasp model (Mode 2) is produced by training/exporting with ultralytics
and is intentionally NOT downloaded here — see the build guide section 5.
"""

from __future__ import annotations

import argparse
import pathlib
import urllib.request

ROOT = pathlib.Path(__file__).resolve().parent.parent
MODELS = ROOT / "models"

# MediaPipe Tasks model bundles (official Google storage).
ASSETS = {
    "gesture_recognizer.task":
        "https://storage.googleapis.com/mediapipe-models/gesture_recognizer/"
        "gesture_recognizer/float16/latest/gesture_recognizer.task",
    # Optional: bare hand-landmarker (gesture recognizer already includes landmarks).
    "hand_landmarker.task":
        "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
        "hand_landmarker/float16/latest/hand_landmarker.task",
}

DEFAULT = ["gesture_recognizer.task"]


def fetch(name: str) -> None:
    url = ASSETS[name]
    dest = MODELS / name
    if dest.exists():
        print(f"  ✓ {name} (already present, {dest.stat().st_size // 1024} KB)")
        return
    print(f"  ↓ {name} …")
    urllib.request.urlretrieve(url, dest)
    print(f"  ✓ {name} ({dest.stat().st_size // 1024} KB)")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true", help="fetch every known asset")
    ap.add_argument("names", nargs="*", choices=list(ASSETS) + [], default=None,
                    help="specific asset(s) to fetch")
    args = ap.parse_args()

    MODELS.mkdir(exist_ok=True)
    wanted = list(ASSETS) if args.all else (args.names or DEFAULT)
    print(f"models -> {MODELS}")
    for name in wanted:
        fetch(name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
