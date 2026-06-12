# ESP32 servo-controller firmware

Hard-real-time controller for the gripper arm. Talks to the Pi over USB-CDC
serial (CRC-framed `J/M/E/H/P`, see `protocol.h` ↔ `gripper/protocol.py`) and
drives the Hiwonder/LewanSoul **LX bus servos** over UART2 (`lx_servo.h`).

## Files
| file | what |
|---|---|
| `esp32_servo.ino` | main loop: parse host frames, slew + write servos, watchdog, e-stop, telemetry |
| `protocol.h` | host↔MCU wire format + CRC-8/MAXIM (mirrors `gripper/protocol.py`) |
| `lx_servo.h` | LX bus-servo packet build/parse (unit-tested in `../test/test_lx_servo.c`) |
| `limits.h` | **generated** from `config/robot.yaml` by `tools/gen_firmware_limits.py` |

## Wiring
- **Bus servos:** ESP32 UART2 `TX=GPIO17`, `RX=GPIO16` → the Hiwonder bus-servo
  bridge/debug board → the single half-duplex servo data line (5 V logic). All
  six servos share the chain; each has a unique ID (set them once with Hiwonder's
  tool: base=6, shoulder=5, elbow=4, wrist=3, grip=1, locked=2).
- **Servo power:** the **separate 7.4 V rail** — never the Pi. The ESP32 is
  USB-powered from the Pi.
- **E-stop button:** normally-closed, wired between `GPIO4` and GND
  (`INPUT_PULLUP`). Intact loop reads LOW = OK; pressed *or a broken wire* reads
  HIGH = stop (fail-safe). Set `HAVE_ESTOP_BUTTON 0` to bench-test without one.
- **Power-cut FET:** low-side N-FET gate on `GPIO5` (HIGH = servo power enabled).

## Build & flash
Arduino IDE / arduino-cli with the ESP32 core. `protocol.h`, `limits.h`,
`lx_servo.h` sit next to the `.ino` so they're picked up automatically.

```bash
# regenerate limits.h whenever robot.yaml changes, THEN reflash:
python3 ../../tools/gen_firmware_limits.py
arduino-cli compile --fqbn esp32:esp32:esp32 .
arduino-cli upload  --fqbn esp32:esp32:esp32 -p /dev/ttyUSB0 .
```

## Bring-up order (Phase 1)
1. **Power + IDs:** confirm every servo answers on its ID with Hiwonder's tool.
2. **Self-test sweep (no Pi):** set `#define SELFTEST_SWEEP 1`, flash — `SERVO_ID[0]`
   (base) sweeps between its calibrated `SERVO_POS_MIN/MAX`. Confirms power,
   wiring, bus, and direction. Set it back to `0` afterwards.
3. **Calibrate:** jog each joint to its mechanical min/max, read `SERVO_POS_READ`,
   and put those LX units into `config/robot.yaml` (`servo_pos_min/max`). Flip a
   joint's pair (or `invert`) if it runs backwards. Regenerate `limits.h`, reflash.
4. **Under Pi control:** flash with `SELFTEST_SWEEP 0`, then from the Pi:
   ```bash
   python3 tools/servo_sweep.py --joint base --port /dev/ttyACM0
   ```
   The base joint should track the sine wave; the rest hold at home. This proves
   the full Pi → CRC frame → ESP32 → LX servo path.

## Notes
- Servo setpoints are written at `SERVO_WRITE_HZ` (50) with `MOVE_TIME_MS` (20)
  so each LX servo interpolates smoothly between steps; the slew limit in
  `JOINT_MAX_STEP_CD` caps speed independently of the host.
- If you use a bare single-wire (transistor) half-duplex circuit instead of a
  bridge board, the read path already tolerates the TX echo, but you may need a
  direction-enable GPIO — add it in `busSend`/`busReadFrame`.
- The `200 ms` command watchdog holds the last pose if the Pi goes quiet; it does
  **not** go limp (it may be holding an object). The hard e-stop cuts power.
