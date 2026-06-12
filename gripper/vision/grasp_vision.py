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
  v3  GG-CNN/GG-CNN2 grasp-quality maps via ONNX Runtime (torch-free on the Pi),
      decoded to the SAME observation schema so the IBVS loop is unchanged.
      Trained by tools/train_ggcnn.py; optional VL53L5CX ToF range (gripper/vision/
      tof.py) gates the plunge. NOTE: the 8x8 ToF is a coarse *range* signal, not a
      dense depth image — RGB GG-CNN is the primary path here, not true RGB-D.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from multiprocessing.synchronize import Event as EventT
from typing import Callable, Optional

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


# ---------------------------------------------------------------------------
# v3 — GG-CNN grasp-quality maps
# ---------------------------------------------------------------------------

@dataclass
class GGCNNConfig:
    """Knobs for the GG-CNN post-processing. ``out_size`` must match the model's
    square input/output side; ``width_scale`` follows the GG-CNN convention that
    the width map is pixels/``width_scale`` (150 in the original net)."""

    out_size: int = 300          # GG-CNN input / output map side (square)
    width_scale: float = 150.0   # width-map units -> pixels in the out_size frame
    q_threshold: float = 0.25    # quality cutoff for the apparent-size proxy
    gaussian_sigma: float = 2.0  # smooth the quality map before argmax
    use_depth: bool = False      # feed a (ToF/RGB-D) depth channel instead of RGB


def _center_crop_box(frame_w: int, frame_h: int, out_size: int) -> tuple[int, int, int]:
    """Square centre-crop GG-CNN preprocessing uses: returns (x0, y0, side)."""
    side = min(frame_w, frame_h)
    return (frame_w - side) // 2, (frame_h - side) // 2, side


def _smooth(a, sigma: float):
    """Separable Gaussian blur (numpy only — GG-CNN smooths quality before argmax)."""
    import numpy as np

    if sigma <= 0:
        return a
    radius = max(1, int(3 * sigma))
    x = np.arange(-radius, radius + 1)
    k = np.exp(-(x * x) / (2 * sigma * sigma))
    k /= k.sum()
    blur = np.apply_along_axis(lambda v: np.convolve(v, k, mode="same"), 1, a)
    return np.apply_along_axis(lambda v: np.convolve(v, k, mode="same"), 0, blur)


def decode_ggcnn(quality, cos2, sin2, width, frame_w: int, frame_h: int,
                 cfg: GGCNNConfig = GGCNNConfig()) -> dict:
    """Decode GG-CNN's four output maps into the grasp observation schema. Pure
    (numpy) — unit-tested. Maps the best-quality pixel back to frame coordinates
    through the same centre-crop/resize the detector applies.

    - ``theta`` from the angle maps: ``0.5*atan2(sin2θ, cos2θ)`` -> (-90°, 90°].
    - ``area_frac`` (the IBVS depth proxy) = fraction of the GG-CNN field whose
      quality exceeds ``q_threshold`` — it grows on approach like a blob area.
    """
    import numpy as np

    q = np.asarray(quality, dtype=np.float32)
    cos2 = np.asarray(cos2, dtype=np.float32)
    sin2 = np.asarray(sin2, dtype=np.float32)
    width = np.asarray(width, dtype=np.float32)

    qf = _smooth(q, cfg.gaussian_sigma)
    out_h, out_w = qf.shape
    row, col = (int(v) for v in np.unravel_index(int(np.argmax(qf)), qf.shape))

    theta_deg = float(np.degrees(0.5 * np.arctan2(sin2[row, col], cos2[row, col])))
    width_map_px = float(width[row, col]) * cfg.width_scale

    x0, y0, side = _center_crop_box(frame_w, frame_h, out_w)
    scale = side / out_w                      # out-map px -> frame px
    cx = x0 + (col + 0.5) * scale
    cy = y0 + (row + 0.5) * scale
    half_w, half_h = frame_w / 2.0, frame_h / 2.0

    return {
        "u_norm": (cx - half_w) / half_w,
        "v_norm": (cy - half_h) / half_h,
        "area_frac": float((qf > cfg.q_threshold).mean()),
        "theta_deg": wrap_grasp_angle(theta_deg),
        "width": (width_map_px * scale) / frame_w,   # jaw width as a frame fraction
        "score": float(qf[row, col]),
    }


class GGCNNGraspDetector:
    """v3: GG-CNN/GG-CNN2 grasp maps -> normalized grasp. Runs the exported model
    on **ONNX Runtime** (no torch needed on the Pi). Assumes the canonical GG-CNN
    output order [quality, cos2θ, sin2θ, width]."""

    def __init__(self, model_path: str, cfg: Optional[GGCNNConfig] = None,
                 providers: Optional[list] = None):
        import onnxruntime as ort

        self.cfg = cfg or GGCNNConfig()
        self.sess = ort.InferenceSession(
            model_path, providers=providers or ["CPUExecutionProvider"])
        self.input_name = self.sess.get_inputs()[0].name

    def _preprocess(self, frame, depth):
        import cv2
        import numpy as np

        h, w = frame.shape[:2]
        x0, y0, side = _center_crop_box(w, h, self.cfg.out_size)
        out = self.cfg.out_size
        if self.cfg.use_depth and depth is not None:
            crop = depth[y0:y0 + side, x0:x0 + side]
            img = cv2.resize(crop.astype("float32"), (out, out))
            img = np.clip(img - img.mean(), -1.0, 1.0)   # GG-CNN zero-centres depth
            return img[None, None].astype(np.float32)
        crop = frame[y0:y0 + side, x0:x0 + side]
        img = cv2.resize(crop, (out, out))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype("float32") / 255.0
        img = img - 0.5                                  # simple centring
        return img.transpose(2, 0, 1)[None].astype(np.float32)

    def detect(self, frame, frame_w: int, frame_h: int, depth=None) -> Optional[dict]:
        import numpy as np

        x = self._preprocess(frame, depth)
        q, cos2, sin2, width = (np.squeeze(o) for o in
                                self.sess.run(None, {self.input_name: x}))
        return decode_ggcnn(q, cos2, sin2, width, frame_w, frame_h, self.cfg)


def make_detector(yolo_path: str, ggcnn_path: str = "",
                  cfg: Optional[GGCNNConfig] = None
                  ) -> Callable[..., Optional[dict]]:
    """Pick the best available detector as ``callable(frame, w, h, depth=None)``:
    GG-CNN (v3) > YOLO (v2) > blob (v1). Falling back keeps the arm working even
    before a model is trained/exported."""
    if ggcnn_path and os.path.exists(ggcnn_path):
        det = GGCNNGraspDetector(ggcnn_path, cfg=cfg)
        return lambda f, w, h, depth=None: det.detect(f, w, h, depth)
    if yolo_path and os.path.exists(yolo_path):
        det = YoloGraspDetector(yolo_path)
        return lambda f, w, h, depth=None: det.detect(f, w, h)
    return lambda f, w, h, depth=None: detect_grasp_v1(f, w, h)


def run(shm_name: str, active: EventT, cam_index: int = 1,
        width: int = 640, height: int = 480, detect_hz: int = 8,
        model_path: str = "models/yolo11n_objects.onnx",
        ggcnn_path: str = "models/ggcnn.onnx", use_tof: bool = False) -> None:
    """Process entry point. Selects GG-CNN > YOLO > blob by which model exists,
    optionally reads a VL53L5CX for plunge-gating range, and publishes a
    normalized grasp obs (with ``range_mm`` when ToF is present) while ``active``
    is set."""
    import cv2

    out = SharedLatest(name=shm_name, create=False)
    detect = make_detector(model_path, ggcnn_path)

    tof = None
    if use_tof:
        from .tof import VL53L5CX
        tof = VL53L5CX()

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
            range_mm = tof.center_distance_mm() if tof is not None else None
            depth = tof.depth_image(w, h) if (tof is not None) else None
            grasp = detect(frame, w, h, depth)
            if grasp is not None:
                grasp["t_capture"] = time.monotonic()
                if range_mm is not None:
                    grasp["range_mm"] = range_mm
                out.put(grasp)
            time.sleep(period)
    finally:
        cap.release()
