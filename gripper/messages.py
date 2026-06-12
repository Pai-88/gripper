"""In-process dataclasses passed between the vision, control and comms tasks.

Field names deliberately echo ROS 2 ``geometry_msgs`` / ``sensor_msgs`` so the
stack could be ported to ROS 2 in roughly a day if ever needed (see the build
guide, "Recommendation: lightweight custom Python, not ROS 2").
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence


@dataclass
class JointTargets:
    """Commanded joint angles in DEGREES, plus grip as an open fraction 0..1.

    This is the single currency the controller speaks; conversion to wire
    centidegrees / milli-units happens in :mod:`gripper.comms.mcu_link`.
    """

    base: float = 0.0
    shoulder: float = 0.0
    elbow: float = 0.0
    wrist: float = 0.0
    grip: float = 1.0  # 1.0 = open, 0.0 = closed

    def joints(self) -> tuple[float, float, float, float]:
        return (self.base, self.shoulder, self.elbow, self.wrist)


@dataclass
class JointState:
    """Feedback read back from the bus servos (potentiometer angle, not true pose)."""

    base: float = 0.0
    shoulder: float = 0.0
    elbow: float = 0.0
    wrist: float = 0.0
    grip: float = 0.0
    currents_mA: tuple = ()
    temps_C: tuple = ()


@dataclass
class HandObservation:
    """Output of the teleop vision process (Mode 1)."""

    landmarks: Sequence  # 21 (x, y, z) tuples in normalized image space
    handedness: str = "Right"
    gesture: Optional[str] = None
    gesture_score: float = 0.0
    t_capture: float = 0.0  # monotonic seconds, stamped at frame grab


@dataclass
class GraspPose:
    """Output of the autonomous grasp vision process (Mode 2): an image-space grasp."""

    u: float  # target pixel x
    v: float  # target pixel y
    theta: float  # grasp angle, radians
    width: float  # required jaw width (normalized)
    score: float = 0.0
    t_capture: float = 0.0


@dataclass
class Telemetry:
    """Decoded ESP32 -> Pi telemetry frame, surfaced to the dashboard/logger."""

    state: str
    joints: tuple
    grip: float
    vbat_mV: int
    flags: int
    seq: int
    t: float = 0.0
