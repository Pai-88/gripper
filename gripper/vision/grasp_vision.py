"""PROCESS B — autonomous grasp vision (Mode 2), Phase 6.

Captures the wrist (eye-in-hand) camera, finds the target, and publishes a
normalized grasp observation for the main process's IBVS loop (Phase 7,
``control/servoing.py``):

    {"u_norm", "v_norm", "area_frac", "theta_deg", "width", "score", "t_capture"}

``u_norm``/``v_norm`` are the target's offset from frame centre in [-1, 1];
``area_frac`` is the apparent-size depth proxy. The normalization (:func:`normalize_grasp`)
is pure and unit-tested; the detectors below wrap it.

Phasing (build guide §5):
  v1  HSV blob + minAreaRect (pure OpenCV, 30 FPS, no model)
  v2  fine-tuned YOLO11n bbox  (OpenVINO on CPU, or Hailo-8L)
  v3  GG-CNN / GR-ConvNet grasp maps (RGB-D + ToF)   [TODO]
"""

from __future__ import annotations

import os
import time
from multiprocessing.synchronize import Event as EventT
from typing import Optional

from ..shared_state import SharedLatest
from ..control.servoing import wrap_grasp_angle


def normalize_grasp(cx: float, cy: float, rect_w: float, rect_h: float,
                    angle_deg: float, frame_w: int, frame_h: int,
                    score: float = 1.0) -> dict:
    """Convert a pixel-space detection into the servo-error schema. Pure."""
    half_w, half_h = frame_w / 2.0, frame_h / 2.0
    return {
        "u_norm": (cx - half_w) / half_w,
        "v_norm": (cy - half_h) / half_h,
        "area_frac": (rect_w * rect_h) / (frame_w * frame_h),
        "theta_deg": wrap_grasp_angle(angle_deg),
        "width": min(rect_w, rect_h) / frame_w,  # jaw width as a frame fraction
        "score": score,
    }


def detect_grasp_v1(frame, frame_w: int, frame_h: int) -> Optional[dict]:
    """v1: HSV-threshold the largest coloured blob -> normalized grasp."""
    import cv2

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # TODO: calibrate these bounds for your target object / mat.
    mask = cv2.inRange(hsv, (35, 80, 60), (85, 255, 255))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, None)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    c = max(contours, key=cv2.contourArea)
    if cv2.contourArea(c) < 200:
        return None
    (cx, cy), (w, h), angle = cv2.minAreaRect(c)
    return normalize_grasp(cx, cy, w, h, angle, frame_w, frame_h, score=1.0)


class YoloGraspDetector:
    """v2: fine-tuned YOLO11n -> highest-confidence bbox -> normalized grasp.

    The grasp angle is coarse (across the box's short side); for a continuous
    angle, run v1's minAreaRect *inside* the YOLO box (hybrid — build guide §5).
    """

    def __init__(self, model_path: str, conf: float = 0.4):
        from ultralytics import YOLO  # lazy
        self.model = YOLO(model_path)
        self.conf = conf

    def detect(self, frame, frame_w: int, frame_h: int) -> Optional[dict]:
        res = self.model(frame, conf=self.conf, verbose=False)[0]
        if res.boxes is None or len(res.boxes) == 0:
            return None
        i = int(res.boxes.conf.argmax())
        x1, y1, x2, y2 = (float(v) for v in res.boxes.xyxy[i].tolist())
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        bw, bh = x2 - x1, y2 - y1
        angle = 0.0 if bw >= bh else 90.0  # jaws across the short side
        return normalize_grasp(cx, cy, bw, bh, angle, frame_w, frame_h,
                               score=float(res.boxes.conf[i]))


def run(shm_name: str, active: EventT, cam_index: int = 1,
        width: int = 640, height: int = 480, detect_hz: int = 8,
        model_path: str = "models/yolo11n_objects.onnx") -> None:
    """Process entry point. Uses YOLO if the model exists, else the v1 blob
    detector. Publishes a normalized grasp obs while ``active`` is set."""
    import cv2

    out = SharedLatest(name=shm_name, create=False)
    detector = YoloGraspDetector(model_path) if os.path.exists(model_path) else None

    cap = cv2.VideoCapture(cam_index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    period = 1.0 / detect_hz
    try:
        while True:
            if not active.is_set():
                time.sleep(0.02)
                continue
            ok, frame = cap.read()
            if not ok:
                continue
            h, w = frame.shape[:2]
            grasp = (detector.detect(frame, w, h) if detector
                     else detect_grasp_v1(frame, w, h))
            if grasp is not None:
                grasp["t_capture"] = time.monotonic()
                out.put(grasp)
            time.sleep(period)
    finally:
        cap.release()
