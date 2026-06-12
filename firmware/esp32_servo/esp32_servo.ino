// esp32_servo.ino — hard-real-time servo controller for the gripper arm.
//
// Responsibilities (the Pi NEVER does timing-critical work):
//   * parse CRC-framed J/M/E/H/P frames from the Pi over USB-CDC serial;
//   * write the LX bus servos at SERVO_WRITE_HZ, slew-limited, with each step
//     interpolated by the servo over MOVE_TIME_MS;
//   * enforce per-joint soft limits from limits.h (independent of the host);
//   * MCU_WATCHDOG_MS command watchdog -> hold last pose (fail safe);
//   * read a normally-closed e-stop button and drive the low-side power-cut FET;
//   * stream S telemetry (real bus-servo voltage + over-temp flag) at ~20 Hz.
//
// limits.h is GENERATED from config/robot.yaml by tools/gen_firmware_limits.py.
// protocol.h mirrors gripper/protocol.py. lx_servo.h is the LX bus protocol
// (unit-tested in firmware/test/test_lx_servo.c).

#include <Arduino.h>
#include "protocol.h"
#include "limits.h"
#include "lx_servo.h"

// ---- build-time options -----------------------------------------------------
#define SELFTEST_SWEEP     0   // 1 = ignore the Pi and sweep SERVO_ID[0] (bring-up)
#define HAVE_ESTOP_BUTTON  1   // 0 = no physical button wired yet (bench testing)
#define OVERTEMP_C         65  // bus-servo over-temperature trip

// ---- pins -------------------------------------------------------------------
static const int PIN_ESTOP_NC   = 4;   // NC button to GND, INPUT_PULLUP: HIGH = pressed/broken
static const int PIN_POWER_FET  = 5;   // low-side N-FET gate: HIGH = servo power enabled
static const int PIN_BUS_RX     = 16;  // hardware UART2 <- bus bridge
static const int PIN_BUS_TX     = 17;  // hardware UART2 -> bus bridge

// ---- control state ----------------------------------------------------------
static int32_t target_cd[JOINT_COUNT];   // latest commanded (centideg offset-from-min)
static int32_t actual_cd[JOINT_COUNT];   // slew-limited output
static int32_t target_grip = GRIP_MAX;   // grip units 0(closed)..1000(open)
static int32_t actual_grip = GRIP_MAX;
static uint8_t mode = MODE_IDLE;
static uint8_t last_seq = 0;
static uint16_t flags = 0;
static uint16_t vbat_mV = 0;
static bool estop_active = false;
static uint32_t last_cmd_ms = 0;

static const uint32_t TELEM_HZ  = 20;
static const uint32_t HEALTH_HZ = 2;   // bus-servo voltage/temperature polling

// ---- serial line assembly ---------------------------------------------------
static char line[96];
static uint8_t line_len = 0;

// ============================ LX bus-servo I/O ===============================
static void busSend(const uint8_t *buf, uint8_t n) {
  Serial2.write(buf, n);
  Serial2.flush();   // ensure the half-duplex line is released before we listen
}

// Read one complete LX frame into out[]. Returns its length, or 0 on timeout.
// Robust to the line echoing our own TX (DIY single-wire circuits).
static uint8_t busReadFrame(uint8_t *out, uint8_t maxn, uint32_t timeout_ms) {
  uint32_t t0 = millis();
  uint8_t idx = 0, need = 0, state = 0;
  while (millis() - t0 < timeout_ms) {
    if (!Serial2.available()) continue;
    uint8_t b = (uint8_t)Serial2.read();
    switch (state) {
      case 0: if (b == LX_HEADER) state = 1; break;
      case 1: state = (b == LX_HEADER) ? 2 : 0; break;
      case 2: out[0] = LX_HEADER; out[1] = LX_HEADER; out[2] = b; idx = 3; state = 3; break;
      case 3: out[idx++] = b; need = (uint8_t)(b - 1); state = 4; break;  // LEN
      default:
        out[idx++] = b;
        if (--need == 0) return idx;
        if (idx >= maxn) return 0;
    }
  }
  return 0;
}

// Send a read request and wait for the matching int16 response value.
static bool busReadValue16(uint8_t id, uint8_t cmd, int16_t *out, uint32_t timeout_ms) {
  uint8_t pkt[8];
  busSend(pkt, lx_build_read(pkt, id, cmd));
  uint32_t t0 = millis();
  uint8_t frame[16];
  while (millis() - t0 < timeout_ms) {
    uint8_t n = busReadFrame(frame, sizeof frame, timeout_ms);
    if (n && lx_parse_value16(frame, n, cmd, NULL, out)) return true;
  }
  return false;
}

static bool busReadValue8(uint8_t id, uint8_t cmd, uint8_t *out, uint32_t timeout_ms) {
  uint8_t pkt[8];
  busSend(pkt, lx_build_read(pkt, id, cmd));
  uint8_t frame[16];
  uint32_t t0 = millis();
  while (millis() - t0 < timeout_ms) {
    uint8_t n = busReadFrame(frame, sizeof frame, timeout_ms);
    if (n && lx_parse_value8(frame, n, cmd, NULL, out)) return true;
  }
  return false;
}

static void busServoMove(uint8_t id, uint16_t pos, uint16_t time_ms) {
  uint8_t pkt[12];
  busSend(pkt, lx_build_move(pkt, id, pos, time_ms));
}

static void busSetTorque(bool on) {
  uint8_t pkt[8];
  for (int i = 0; i < JOINT_COUNT; i++) busSend(pkt, lx_build_load(pkt, SERVO_ID[i], on));
  busSend(pkt, lx_build_load(pkt, GRIP_SERVO_ID, on));
}

// ============================ control ========================================
static uint16_t map_units(int32_t v, int32_t in_lo, int32_t in_hi,
                          int32_t out_lo, int32_t out_hi) {
  if (in_hi == in_lo) return (uint16_t)out_lo;
  if (v < in_lo) v = in_lo;
  if (v > in_hi) v = in_hi;
  long pos = out_lo + (long)(out_hi - out_lo) * (v - in_lo) / (in_hi - in_lo);
  if (pos < LX_POS_MIN_UNIT) pos = LX_POS_MIN_UNIT;
  if (pos > LX_POS_MAX_UNIT) pos = LX_POS_MAX_UNIT;
  return (uint16_t)pos;
}

static int32_t slew_to(int32_t cur, int32_t tgt, int32_t max_step) {
  int32_t d = tgt - cur;
  if (d >  max_step) return cur + max_step;
  if (d < -max_step) return cur - max_step;
  return tgt;
}

static void enable_power(bool on) { digitalWrite(PIN_POWER_FET, on ? HIGH : LOW); }

static bool estop_pressed() {
#if HAVE_ESTOP_BUTTON
  return digitalRead(PIN_ESTOP_NC) == HIGH;  // NC to GND: HIGH = pressed or wire broken
#else
  return false;
#endif
}

static void go_estop() {
  estop_active = true;
  mode = MODE_IDLE;
  flags |= FLAG_WATCHDOG;
  enable_power(false);   // hard low-side cut
  busSetTorque(false);   // belt-and-braces (line may still be powered downstream)
}

// Re-arm after an e-stop. Triggered by the next valid M/H frame from the Pi.
static void clear_estop() {
  if (!estop_active) return;
  estop_active = false;
  flags &= ~FLAG_WATCHDOG;
  enable_power(true);
  busSetTorque(true);
}

static void drive_servos() {
  if (estop_active) return;
  for (int i = 0; i < JOINT_COUNT; i++) {
    actual_cd[i] = slew_to(actual_cd[i], target_cd[i], JOINT_MAX_STEP_CD[i]);
    uint16_t pos = map_units(actual_cd[i], 0, JOINT_SPAN_CD[i],
                             SERVO_POS_MIN[i], SERVO_POS_MAX[i]);
    busServoMove(SERVO_ID[i], pos, MOVE_TIME_MS);
  }
  actual_grip = slew_to(actual_grip, target_grip, GRIP_MAX_STEP);
  uint16_t gpos = map_units(actual_grip, 0, 1000, GRIP_POS_CLOSED, GRIP_POS_OPEN);
  busServoMove(GRIP_SERVO_ID, gpos, MOVE_TIME_MS);
}

// Poll one servo's voltage + temperature per call (round-robin, low bus load).
static void poll_health() {
  static uint8_t rr = 0;
  int16_t v;
  if (busReadValue16(SERVO_ID[0], LX_VIN_READ, &v, 8)) {
    vbat_mV = (uint16_t)v;
    if (vbat_mV < 6000) flags |= FLAG_UNDERVOLT; else flags &= ~FLAG_UNDERVOLT;
  }
  uint8_t t;
  if (busReadValue8(SERVO_ID[rr], LX_TEMP_READ, &t, 8)) {
    if (t >= OVERTEMP_C) flags |= FLAG_OVERTEMP; else flags &= ~FLAG_OVERTEMP;
  }
  rr = (uint8_t)((rr + 1) % JOINT_COUNT);
}

// ============================ host link ======================================
static bool handle_line(char *s, uint8_t n) {
  if (n == 0) return false;
  if (n == 1 && s[0] == 'E') { go_estop(); return true; }
  if (n == 1 && s[0] == 'P') { return true; }
  if (n == 1 && s[0] == 'H') {
    clear_estop();
    for (int i = 0; i < JOINT_COUNT; i++) target_cd[i] = HOME_CD[i];
    target_grip = GRIP_MAX;
    last_cmd_ms = millis();
    return true;
  }
  if (s[0] == 'M') { clear_estop(); mode = (uint8_t)atoi(s + 2); last_cmd_ms = millis(); return true; }

  if (s[0] == 'J') {
    int last_comma = -1;
    for (int i = n - 1; i >= 0; i--) { if (s[i] == ',') { last_comma = i; break; } }
    if (last_comma < 0) return false;
    uint8_t want = (uint8_t)atoi(s + last_comma + 1);
    if (crc8_maxim((const uint8_t *)s, last_comma) != want) return false;  // drop bad CRC

    int idx = 0;
    char *tok = strtok(s, ",");          // "J"
    tok = strtok(NULL, ",");
    while (tok && idx < JOINT_COUNT) {
      int32_t v = atol(tok);
      if (v < ANGLE_MIN_CD) v = ANGLE_MIN_CD;
      if (v > JOINT_SPAN_CD[idx]) { v = JOINT_SPAN_CD[idx]; flags |= FLAG_LIMIT_CLAMP; }
      else { flags &= ~FLAG_LIMIT_CLAMP; }
      target_cd[idx++] = v;
      tok = strtok(NULL, ",");
    }
    if (tok) { int32_t gv = atol(tok); target_grip = gv < GRIP_MIN ? GRIP_MIN
                                       : (gv > GRIP_MAX ? GRIP_MAX : gv); tok = strtok(NULL, ","); }
    if (tok) { last_seq = (uint8_t)atoi(tok); }
    last_cmd_ms = millis();
    return true;
  }
  return false;
}

static void pump_serial() {
  while (Serial.available()) {
    char c = (char)Serial.read();
    if (c == '\n') { line[line_len] = 0; handle_line(line, line_len); line_len = 0; }
    else if (line_len < sizeof(line) - 1) { line[line_len++] = c; }
    else { line_len = 0; }  // overflow -> drop
  }
}

static void send_telemetry() {
  char body[96];
  int state_code = estop_active ? 5 : ((flags & FLAG_WATCHDOG) ? 0 : mode);
  int len = snprintf(body, sizeof body, "S,%d,%ld,%ld,%ld,%ld,%ld,%u,%u,%u",
                     state_code, (long)actual_cd[0], (long)actual_cd[1],
                     (long)actual_cd[2], (long)actual_cd[3], (long)actual_grip,
                     vbat_mV, flags, last_seq);
  uint8_t crc = crc8_maxim((const uint8_t *)body, len);
  Serial.printf("%s,%u\n", body, crc);
}

// ============================ Arduino entry ==================================
void setup() {
  Serial.begin(1000000);
  Serial2.begin(115200, SERIAL_8N1, PIN_BUS_RX, PIN_BUS_TX);
  pinMode(PIN_ESTOP_NC, INPUT_PULLUP);
  pinMode(PIN_POWER_FET, OUTPUT);
  enable_power(true);
  for (int i = 0; i < JOINT_COUNT; i++) { target_cd[i] = HOME_CD[i]; actual_cd[i] = HOME_CD[i]; }
  busSetTorque(true);
  last_cmd_ms = millis();
}

#if SELFTEST_SWEEP
// Bring-up: sweep SERVO_ID[0] between its calibrated limits, no Pi needed.
void loop() {
  uint16_t lo = (uint16_t)SERVO_POS_MIN[0], hi = (uint16_t)SERVO_POS_MAX[0];
  uint32_t t = millis() % 4000;
  float ph = t < 2000 ? t / 2000.0f : (4000 - t) / 2000.0f;  // triangle 0..1
  busServoMove(SERVO_ID[0], (uint16_t)(lo + ph * (hi - lo)), 40);
  delay(20);
}
#else
void loop() {
  static uint32_t next_servo = 0, next_telem = 0, next_health = 0;
  uint32_t now = millis();

  pump_serial();

  if (estop_pressed() && !estop_active) go_estop();

  if (now - last_cmd_ms > MCU_WATCHDOG_MS) flags |= FLAG_WATCHDOG;  // host quiet: hold
  else if (!estop_active) flags &= ~FLAG_WATCHDOG;

  if (now >= next_servo)  { next_servo  = now + 1000 / SERVO_WRITE_HZ; drive_servos(); }
  if (now >= next_health) { next_health = now + 1000 / HEALTH_HZ; if (!estop_active) poll_health(); }
  if (now >= next_telem)  { next_telem  = now + 1000 / TELEM_HZ; send_telemetry(); }
}
#endif
