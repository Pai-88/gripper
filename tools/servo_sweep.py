#!/usr/bin/env python3
"""Phase-1 bring-up: sweep ONE joint sinusoidally under Pi control by streaming
full J frames to the ESP32 (the swept joint moves, the rest hold at home).

    python3 tools/servo_sweep.py --joint base --port /dev/ttyACM0
    python3 tools/servo_sweep.py --joint wrist --amp 30 --period 2 --dry   # no hardware

Reads config/robot.yaml directly (only needs pyyaml; pyserial when not --dry) and
reuses the tested gripper.protocol / mcu_link conversion so the wire bytes match
the firmware exactly.
"""

from __future__ import annotations

import argparse
import math
import pathlib
import sys
import time
from types import SimpleNamespace

import yaml

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from gripper import JOINT_NAMES, protocol  # noqa: E402
from gripper.comms.mcu_link import targets_to_wire  # noqa: E402
from gripper.messages import JointTargets  # noqa: E402

try:
    import serial  # pyserial
except ImportError:
    serial = None


def load_joints() -> tuple[dict, dict]:
    cfg = yaml.safe_load(open(ROOT / "config" / "robot.yaml"))
    joints = {}
    for name, j in cfg["joints"].items():
        joints[name] = SimpleNamespace(
            min_deg=j["min_deg"], max_deg=j["max_deg"], home_deg=j["home_deg"],
            invert=j.get("invert", False), span_deg=j["max_deg"] - j["min_deg"],
        )
    return joints, cfg["serial"]


def home_targets(joints: dict) -> JointTargets:
    return JointTargets(
        base=joints["base"].home_deg, shoulder=joints["shoulder"].home_deg,
        elbow=joints["elbow"].home_deg, wrist=joints["wrist"].home_deg,
        grip=joints["grip"].home_deg / 100.0,
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--joint", choices=JOINT_NAMES, default="base")
    ap.add_argument("--port", default=None, help="serial port (default: robot.yaml)")
    ap.add_argument("--amp", type=float, default=20.0, help="sweep amplitude, deg")
    ap.add_argument("--period", type=float, default=3.0, help="sweep period, s")
    ap.add_argument("--rate", type=int, default=50, help="frame send rate, Hz")
    ap.add_argument("--duration", type=float, default=10.0, help="run time, s (0 = forever)")
    ap.add_argument("--dry", action="store_true", help="print frames, no serial")
    args = ap.parse_args()

    joints, scfg = load_joints()
    j = joints[args.joint]
    center = j.home_deg
    lo, hi = j.min_deg, j.max_deg
    port = args.port or scfg["port"]
    baud = scfg["baud"]

    ser = None
    if not args.dry:
        if serial is None:
            print("pyserial not installed: pip install pyserial (or use --dry)", file=sys.stderr)
            return 2
        ser = serial.Serial(port, baud, timeout=0)
        ser.write(protocol.encode_mode(protocol.MODE_TELEOP))  # clear e-stop / arm

    print(f"sweeping {args.joint}: {center}±{args.amp}° "
          f"(limits {lo}..{hi}) at {args.rate} Hz -> {'DRY' if args.dry else port}")

    seq = 0
    period_s = 1.0 / args.rate
    t0 = time.monotonic()
    try:
        while True:
            t = time.monotonic() - t0
            if args.duration and t > args.duration:
                break
            angle = center + args.amp * math.sin(2 * math.pi * t / args.period)
            angle = max(lo, min(hi, angle))
            tgt = home_targets(joints)
            setattr(tgt, args.joint, angle)
            cd, grip = targets_to_wire(tgt, joints)
            seq = (seq + 1) & 0xFF
            frame = protocol.encode_joint(cd, grip, seq)
            if args.dry:
                if seq % 10 == 0:
                    print(f"  t={t:4.1f}s {args.joint}={angle:6.1f}°  {frame!r}")
            else:
                ser.write(frame)
            time.sleep(period_s)
    except KeyboardInterrupt:
        pass
    finally:
        if ser is not None:
            ser.write(protocol.encode_home())  # park at home on exit
            ser.close()
    print("done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
