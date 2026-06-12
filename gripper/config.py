"""Typed config loader. Reads ``config/robot.yaml`` into validated Pydantic
models so the rest of the code never touches magic numbers or raw dicts."""

from __future__ import annotations

import pathlib
from typing import Optional

import yaml
from pydantic import BaseModel


class JointCfg(BaseModel):
    servo_id: int
    min_deg: float
    max_deg: float
    home_deg: float
    max_rate_dps: float
    invert: bool = False

    @property
    def span_deg(self) -> float:
        return self.max_deg - self.min_deg


class SerialCfg(BaseModel):
    port: str
    baud: int
    command_hz: int = 50
    telemetry_hz: int = 20


class GeometryCfg(BaseModel):
    l1_m: float
    l2_m: float


class CameraCfg(BaseModel):
    index: int
    width: int = 640
    height: int = 480
    fps: int = 30


class OneEuroCfg(BaseModel):
    freq: float = 30.0
    mincutoff: float = 1.0
    beta: float = 0.007
    dcutoff: float = 1.0


class GestureCfg(BaseModel):
    hold_frames: int = 12
    min_score: float = 0.7


class TeleopCfg(BaseModel):
    model_path: str = "models/gesture_recognizer.task"
    mirror: bool = True
    one_euro: OneEuroCfg = OneEuroCfg()
    gesture: GestureCfg = GestureCfg()


class WatchdogCfg(BaseModel):
    mcu_timeout_ms: int = 200
    vision_stale_ms: int = 300


class GraspCfg(BaseModel):
    model: str = "models/yolo11n_objects.onnx"
    detect_hz: int = 8
    servo_gain_x: float = 0.4
    servo_gain_scale: float = 0.6
    commit_frames: int = 5


class Config(BaseModel):
    serial: SerialCfg
    joints: dict[str, JointCfg]
    locked_servo_id: int
    geometry: GeometryCfg
    cameras: dict[str, CameraCfg]
    teleop: TeleopCfg = TeleopCfg()
    watchdog: WatchdogCfg = WatchdogCfg()
    grasp: GraspCfg = GraspCfg()


DEFAULT_PATH = pathlib.Path(__file__).resolve().parent.parent / "config" / "robot.yaml"


def load_config(path: Optional[str | pathlib.Path] = None) -> Config:
    path = pathlib.Path(path) if path else DEFAULT_PATH
    with open(path) as f:
        data = yaml.safe_load(f)
    return Config(**data)
