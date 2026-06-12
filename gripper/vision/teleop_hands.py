"""PROCESS A — teleop hand tracking (Mode 1).

Captures the fixed webcam and runs a single MediaPipe **Gesture Recognizer** in
LIVE_STREAM mode, which returns BOTH the 21 hand landmarks AND the canned gesture
(Open_Palm / Victory / Thumb_Up / Closed_Fist) from one model — so the Pi runs
one network, not two. The latest :class:`HandObservation` is published into a
:class:`SharedLatest` channel for the main control process.

Heavy deps (mediapipe, opencv) are imported lazily inside ``run`` / ``main`` so
this module imports anywhere; the result-parsing (:func:`result_to_observation`)
is a pure function and unit-tested in ``tests/test_teleop_hands.py``. The
landmark->joint mapping lives in ``control/teleop_map.py`` (also tested).
"""

from __future__ import annotations

import time
from multiprocessing.synchronize import Event as EventT
from typing import Any, Optional

from ..shared_state import SharedLatest

# Gestures MediaPipe emits for "no gesture" — treated as None.
_NULL_GESTURE = {"", "None"}


def result_to_observation(result: Any, t_capture: float) -> dict:
    """Convert a MediaPipe GestureRecognizerResult (duck-typed) into the
    HandObservation dict we publish. Call only when ``result.hand_landmarks``
    is non-empty. Pure — no MediaPipe import needed to run or test it."""
    landmarks = [(p.x, p.y, p.z) for p in result.hand_landmarks[0]]

    handedness = "Right"
    if getattr(result, "handedness", None) and result.handedness[0]:
        handedness = result.handedness[0][0].category_name

    gesture: Optional[str] = None
    score = 0.0
    if getattr(result, "gestures", None) and result.gestures[0]:
        top = result.gestures[0][0]
        name = top.category_name
        if name and name not in _NULL_GESTURE:
            gesture, score = name, float(top.score)

    return {
        "landmarks": landmarks,
        "handedness": handedness,
        "gesture": gesture,
        "gesture_score": score,
        "t_capture": t_capture,
    }


def run(shm_name: str, active: EventT, cam_index: int = 0,
        width: int = 640, height: int = 480,
        model_path: str = "models/gesture_recognizer.task",
        mirror: bool = True) -> None:
    """Process entry point. Streams HandObservations into the named channel
    while ``active`` is set."""
    import os

    import cv2
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision as mp_vision

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"{model_path} missing — run: python3 tools/download_models.py")

    out = SharedLatest(name=shm_name, create=False)

    latest: dict[str, Any] = {"result": None}

    def on_result(result, output_image, timestamp_ms):  # LIVE_STREAM callback
        latest["result"] = result

    options = mp_vision.GestureRecognizerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=model_path),
        running_mode=mp_vision.RunningMode.LIVE_STREAM,
        num_hands=1,
        result_callback=on_result,
    )
    recognizer = mp_vision.GestureRecognizer.create_from_options(options)

    cap = cv2.VideoCapture(cam_index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # always grab the freshest frame

    t0 = time.monotonic()
    try:
        while True:
            if not active.is_set():
                time.sleep(0.02)
                continue
            ok, frame = cap.read()
            if not ok:
                continue
            if mirror:
                frame = cv2.flip(frame, 1)  # selfie view: hand-left -> arm-left
            t_capture = time.monotonic()

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            recognizer.recognize_async(mp_image, int((t_capture - t0) * 1000))

            result = latest["result"]
            if result is not None and result.hand_landmarks:
                out.put(result_to_observation(result, t_capture))
    finally:
        cap.release()
        recognizer.close()


def main() -> None:
    """Standalone bench test: open the camera, print FPS + latest gesture, and
    (with --preview) draw landmarks. Verifies hand tracking before the arm.

        python3 -m gripper.vision.teleop_hands --preview
    """
    import argparse
    import os

    import cv2
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision as mp_vision

    ap = argparse.ArgumentParser()
    ap.add_argument("--cam", type=int, default=0)
    ap.add_argument("--model", default="models/gesture_recognizer.task")
    ap.add_argument("--preview", action="store_true")
    args = ap.parse_args()
    if not os.path.exists(args.model):
        raise SystemExit(f"{args.model} missing — run tools/download_models.py")

    latest: dict[str, Any] = {"result": None}
    options = mp_vision.GestureRecognizerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=args.model),
        running_mode=mp_vision.RunningMode.LIVE_STREAM, num_hands=1,
        result_callback=lambda r, img, ts: latest.__setitem__("result", r),
    )
    recognizer = mp_vision.GestureRecognizer.create_from_options(options)
    cap = cv2.VideoCapture(args.cam)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    t0 = time.monotonic()
    frames = 0
    last_report = t0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                continue
            frame = cv2.flip(frame, 1)
            now = time.monotonic()
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            recognizer.recognize_async(
                mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb),
                int((now - t0) * 1000))
            frames += 1

            result = latest["result"]
            if result is not None and result.hand_landmarks:
                obs = result_to_observation(result, now)
                if args.preview:
                    h, w = frame.shape[:2]
                    for (x, y, _z) in obs["landmarks"]:
                        cv2.circle(frame, (int(x * w), int(y * h)), 3, (0, 255, 0), -1)
                    cv2.putText(frame, str(obs["gesture"]), (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            if now - last_report >= 1.0:
                g = latest["result"]
                gesture = (result_to_observation(g, now)["gesture"]
                           if g and g.hand_landmarks else "—")
                print(f"{frames/(now-last_report):4.1f} fps  gesture={gesture}")
                frames = 0
                last_report = now
            if args.preview:
                cv2.imshow("teleop_hands", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    finally:
        cap.release()
        recognizer.close()
        if args.preview:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
