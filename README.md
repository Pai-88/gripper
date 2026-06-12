# gripper

A dual-mode, camera-controlled **4-DOF robotic gripper** running on a **Raspberry Pi 5 + ESP32**.

Two modes share one arm, switched by a hand gesture:

- **Mode 1 — Teleop:** a fixed webcam watches your hand; MediaPipe estimates hand pose; the arm mimics it live. *(Assistive / prosthetic-telemanipulation flavour.)*
- **Mode 2 — Autonomous grasp:** a wrist-mounted ("eye-in-hand") camera sees an object; the arm aligns and picks it up by visual servoing. You supervise and confirm the pick with a gesture *(shared autonomy)*.

> Full design rationale, bill of materials, power budget, timeline and the BME/ML framing are in **[`../gripper_build_guide.pdf`](../gripper_build_guide.pdf)**.

## Architecture

```
hand → USB webcam ─┐                          ┌─ Pi 5: perception / ML (this package)
object → wrist cam ─┴→ Raspberry Pi 5 ────────┤    MediaPipe Hands · YOLO/grasp · IBVS
                       │  USB-CDC 1 Mbaud      │    state machine · controller · watchdog
                       │  CRC-framed J/M/E/H/P │
                       ▼                       └─ ESP32: hard real-time (firmware/)
                     ESP32 ──→ 6× LX bus servos     200 Hz slew · soft limits · 200 ms
                       ▲          (4 DOF + grip)     watchdog · low-side e-stop FET
              separate 7.4 V servo rail (never off the Pi)
```

**Why this split:** the Pi does soft-real-time vision/ML; the ESP32 does deterministic servo control and safety, so motion stays smooth and the arm fails safe even if Python hangs or the USB cable is pulled.

## Layout

```
gripper/            python package (Pi side)
  protocol.py         serial wire format + CRC-8/MAXIM  ← single source of truth
  messages.py         dataclasses passed between tasks
  kinematics.py       closed-form 2-link IK / FK
  state_machine.py    IDLE→TELEOP→ARM_AUTO→AUTO_GRASP→RETURN, +ESTOP
  config.py           typed loader for config/robot.yaml
  shared_state.py     seqlock shared-memory channels (vision ⇒ control)
  main.py             asyncio integration glue + `--dry-run`
  control/            limits, teleop_map (One-Euro), kinematics glue, controller
  vision/             gestures (tested) + MediaPipe / YOLO process skeletons
  comms/              async serial link to the ESP32
  safety/             Pi-side watchdog
  telemetry/          JSONL logger + FastAPI dashboard
firmware/esp32_servo/ esp32_servo.ino, protocol.h (mirrors protocol.py), limits.h (generated)
config/               robot.yaml (edit this), calibration.yaml (generated)
tools/                gen_firmware_limits.py, calibrate.py, replay.py
tests/                pure-logic unit tests (run green today)
run_tests.py          zero-dependency test runner
```

## Quickstart

```bash
# 1. Run the unit suite — no third-party deps needed for the core.
python3 run_tests.py            # or: pip install -e '.[dev]' && pytest

# 2. Regenerate the firmware limits header from robot.yaml.
python3 tools/gen_firmware_limits.py

# 3. Exercise the state machine end-to-end with no hardware
#    (needs the lightweight deps: pip install pyyaml pydantic — but no cameras/servos).
python3 -m gripper.main --dry-run

# 4. On the Pi: install the full stack, fetch models, bench-test hand tracking (no arm).
pip install -e .                # mediapipe, opencv, pyserial-asyncio, fastapi, …
python3 tools/download_models.py                  # MediaPipe gesture recognizer
python3 -m gripper.vision.teleop_hands --preview  # live hand landmarks + gesture

# 5. Bring up one servo under Pi control, then the full run (+ live dashboard).
python3 tools/servo_sweep.py --joint base --port /dev/ttyACM0
python3 -m gripper.main --dashboard          # telemetry UI at http://<pi>:8000
python3 -m gripper.telemetry.dashboard       # dashboard demo, standalone
```

Edit limits/geometry in **`config/robot.yaml`**, then re-run `tools/gen_firmware_limits.py` and reflash the ESP32 so both sides agree.

## The Pi ↔ ESP32 contract

Joint angles are sent as **centidegrees offset from each joint's `min_deg`** (always unsigned, 0–18000). The mapping to LX bus-servo units lives in exactly two mirrored places: `comms/mcu_link.py` (Pi) and `firmware/.../limits.h` (MCU). The `J`/`S` frames carry a **CRC-8/MAXIM** checksum implemented identically in `protocol.py` and `protocol.h` and pinned by `tests/test_protocol.py` (`crc8_maxim("123456789") == 0xA1`).

## Build phases

| Phase | Milestone | Status in this scaffold |
|---|---|---|
| 1 | Pi→ESP32 link + one servo | ✅ LX protocol + servo writes (needs flash + calibration) |
| 2 | Full arm under control | controller + limits ✅ |
| 3 | Kinematics | 2-link IK/FK ✅ tested |
| 4 | Teleop MVP | mapping + One-Euro + **MediaPipe capture ✅** (needs a camera) |
| 5 | State machine + safety + dashboard | **✅** FSM + safe-state guard + e-stop wired into the loop; latency p50/p95; live FastAPI dashboard |
| 6 | Object detection | **✅** blob (v1) + lazy YOLO11n (v2) -> normalized grasp obs; train/export the model |
| 7 | Autonomous grasp | **✅** IBVS servo law + grasp sequencer (servo→commit→plunge→close→lift), wired into AUTO_GRASP |

## Safety

Defense in depth: the Pi clamps + slew-limits every command; the **ESP32 re-clamps independently**, runs a 200 ms command watchdog, reads a **normally-closed e-stop button**, and cuts servo power through a **low-side N-FET**. The gesture e-stop (`Open_Palm`) is a convenience layer, never the sole mechanism. Servos run on a **separate 7.4 V rail** — never off the Pi.
