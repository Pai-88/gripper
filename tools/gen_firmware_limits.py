#!/usr/bin/env python3
"""Generate firmware/esp32_servo/limits.h from config/robot.yaml.

Single source of truth for joint limits: run this whenever robot.yaml changes so
the Pi and the ESP32 can never disagree. The generated header is intentionally
plain C arrays (no timestamp) so regeneration produces clean diffs.

    python3 tools/gen_firmware_limits.py
"""

from __future__ import annotations

import pathlib
import sys

import yaml

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from gripper import JOINT_NAMES  # noqa: E402

# Servo setpoints are written at this rate; the LX servos interpolate over
# MOVE_TIME_MS between setpoints. The main firmware loop still runs faster
# (serial/e-stop/watchdog). Slew steps are sized for this rate.
SERVO_WRITE_HZ = 50
LX_UNITS_PER_DEG = 1000.0 / 240.0


def cd(deg: float) -> int:
    return int(round(deg * 100))


def servo_pos_bounds(j: dict) -> tuple[int, int]:
    """LX units at the joint's min/max. Explicit calibration if present, else a
    span-centred default (assumes ~direct drive, servo centred at joint mid)."""
    span = j["max_deg"] - j["min_deg"]
    units = span * LX_UNITS_PER_DEG
    pmin = j.get("servo_pos_min", round(500 - units / 2))
    pmax = j.get("servo_pos_max", round(500 + units / 2))
    clamp = lambda v: int(max(0, min(1000, v)))
    return clamp(pmin), clamp(pmax)


def main() -> int:
    cfg = yaml.safe_load(open(ROOT / "config" / "robot.yaml"))
    joints = cfg["joints"]
    wd_ms = int(cfg["watchdog"]["mcu_timeout_ms"])
    g = joints["grip"]

    def arr(name, ctype, values):
        body = ", ".join(str(v) for v in values)
        return f"static const {ctype} {name}[JOINT_COUNT] = {{ {body} }};"

    span_cd = [cd(joints[n]["max_deg"] - joints[n]["min_deg"]) for n in JOINT_NAMES]
    home_cd = [max(0, cd(joints[n]["home_deg"] - joints[n]["min_deg"])) for n in JOINT_NAMES]
    min_deg = [float(joints[n]["min_deg"]) for n in JOINT_NAMES]
    servo_id = [int(joints[n]["servo_id"]) for n in JOINT_NAMES]
    max_step = [max(1, int(round(joints[n]["max_rate_dps"] * 100 / SERVO_WRITE_HZ)))
                for n in JOINT_NAMES]
    pos_min, pos_max = zip(*(servo_pos_bounds(joints[n]) for n in JOINT_NAMES))
    grip_step = max(1, int(round(g["max_rate_dps"] * 1000 /
                                 ((g["max_deg"] - g["min_deg"]) * SERVO_WRITE_HZ))))
    grip_closed, grip_open = servo_pos_bounds(g)
    move_time_ms = round(1000 / SERVO_WRITE_HZ)

    lines = [
        "// limits.h — GENERATED from config/robot.yaml by tools/gen_firmware_limits.py.",
        "// DO NOT EDIT BY HAND. Regenerate after changing robot.yaml.",
        f"// joint order: {', '.join(JOINT_NAMES)}",
        "#ifndef GRIPPER_LIMITS_H",
        "#define GRIPPER_LIMITS_H",
        "",
        '#include "protocol.h"',
        "",
        f"#define MCU_WATCHDOG_MS {wd_ms}",
        f"#define SERVO_WRITE_HZ  {SERVO_WRITE_HZ}",
        f"#define MOVE_TIME_MS    {move_time_ms}",
        f"#define GRIP_SERVO_ID   {int(g['servo_id'])}",
        f"#define GRIP_MAX_STEP   {grip_step}",
        f"#define GRIP_POS_CLOSED {grip_closed}",
        f"#define GRIP_POS_OPEN   {grip_open}",
        "",
        arr("SERVO_ID", "uint8_t", servo_id),
        arr("JOINT_SPAN_CD", "int32_t", span_cd),
        arr("HOME_CD", "int32_t", home_cd),
        arr("JOINT_MAX_STEP_CD", "int32_t", max_step),
        arr("JOINT_MIN_DEG", "float", [f"{v}f" for v in min_deg]),
        arr("SERVO_POS_MIN", "int32_t", list(pos_min)),
        arr("SERVO_POS_MAX", "int32_t", list(pos_max)),
        "",
        "#endif  // GRIPPER_LIMITS_H",
        "",
    ]
    out = ROOT / "firmware" / "esp32_servo" / "limits.h"
    out.write_text("\n".join(lines))
    print(f"wrote {out.relative_to(ROOT)}  ({len(JOINT_NAMES)} joints, "
          f"watchdog {wd_ms} ms)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
