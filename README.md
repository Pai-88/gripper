# gripper

Control software for a 4-DOF robotic arm (Hiwonder xArm 1S bus servos) with two modes, switched by a hand gesture:

- **Teleop.** A fixed webcam tracks your hand with MediaPipe and the arm follows it.
- **Autonomous grasp.** A wrist camera finds an object and the arm aligns to it by image-based visual servoing (IBVS), then picks it up once you confirm with a gesture.

A Raspberry Pi 5 runs perception and planning. An ESP32 runs servo control and safety, so the arm stops safely if the Python side hangs or the USB cable is pulled.

## Status

The software is written and unit-tested (100 tests, `python3 run_tests.py`), and the state machine runs end to end in `--dry-run` with no hardware. **It has not been run on the arm yet.** Still to do on hardware: flash the ESP32, calibrate each servo's limits, capture the hand-eye transform, and train the grasp models.

## Architecture

```
hand   -> USB webcam   --+
object -> wrist camera --+--> Raspberry Pi 5  (this package)
                                MediaPipe hands, YOLO / GG-CNN grasp,
                                IBVS, state machine, controller, watchdog
                                     |
                                     |  USB-CDC 1 Mbaud, CRC-framed messages
                                     v
                              ESP32  (firmware/)
                                200 Hz slew limit, soft joint limits,
                                200 ms command watchdog, e-stop MOSFET
                                     |
                                     v
                              6 LX bus servos (4 DOF + grip)
                              on a separate 7.4 V rail, never powered
                              from the Pi
```

## Layout

```
gripper/               Python package (Pi side)
  protocol.py          serial wire format + CRC-8/MAXIM (source of truth)
  kinematics.py        closed-form 2-link IK / FK
  state_machine.py     IDLE, TELEOP, ARM_AUTO, AUTO_GRASP, RETURN, ESTOP
  control/             joint limits, teleop mapping (One-Euro), controller
  vision/              gestures, hand tracking, grasp detection
                       (blob, YOLO11n, GG-CNN)
  comms/               async serial link to the ESP32
  safety/              Pi-side watchdog
  telemetry/           JSONL logger + FastAPI dashboard
firmware/esp32_servo/  ESP32 firmware; protocol.h mirrors protocol.py,
                       limits.h is generated
config/                robot.yaml (edit this), calibration.yaml (generated)
tools/                 limit generator, calibration, replay, dataset prep,
                       training
tests/                 unit tests
```

## Running it

```bash
python3 run_tests.py                   # unit tests, no dependencies
python3 tools/gen_firmware_limits.py   # ESP32 limits header from robot.yaml
python3 -m gripper.main --dry-run      # state machine, no cameras or servos
```

On the Pi, with the full dependencies:

```bash
pip install -e .
python3 tools/download_models.py                    # MediaPipe gesture recogniser
python3 -m gripper.vision.teleop_hands --preview    # hand tracking only, no arm
python3 tools/servo_sweep.py --joint base --port /dev/ttyACM0
python3 -m gripper.main --dashboard                 # dashboard at http://<pi>:8000
```

After editing limits or geometry in `config/robot.yaml`, re-run `tools/gen_firmware_limits.py` and reflash the ESP32 so both sides agree.

## Pi to ESP32 protocol

Joint angles travel as unsigned centidegrees offset from each joint's `min_deg` (0 to 18000). The conversion to servo units exists in exactly two mirrored places, `comms/mcu_link.py` and `firmware/.../limits.h`. Command frames carry a CRC-8/MAXIM checksum implemented identically in `protocol.py` and `protocol.h`, and `tests/test_protocol.py` pins it (`crc8_maxim("123456789") == 0xA1`).

## Safety

The Pi clamps and slew-limits every command, and the ESP32 clamps again independently. The ESP32 also runs a 200 ms command watchdog, reads a normally-closed e-stop button, and cuts servo power through a low-side MOSFET. The open-palm gesture stop is a convenience only, never the sole stop.
