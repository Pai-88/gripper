"""gripper — dual-mode (teleop + autonomous) 4-DOF robotic gripper control stack.

Pi 5 = perception / ML brain (this package). ESP32 = hard-real-time servo
controller (see ``firmware/esp32_servo``). The two talk over a CRC-framed serial
link defined once in :mod:`gripper.protocol` and mirrored in
``firmware/esp32_servo/protocol.h``.

Full design rationale: ``../gripper_build_guide.pdf``.
"""

__version__ = "0.1.0"

# The four commanded degrees of freedom, in wire order. The xArm 1S has a fifth
# arm joint that is locked/folded into the shoulder command, plus the gripper.
JOINT_NAMES = ("base", "shoulder", "elbow", "wrist")
JOINT_COUNT = len(JOINT_NAMES)
