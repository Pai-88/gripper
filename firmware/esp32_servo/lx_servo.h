// lx_servo.h — Hiwonder / LewanSoul LX serial bus-servo protocol (LX-15D, LX-225,
// LX-16A, LX-224 …). Pure C (stdint only, NO Arduino) so it is unit-testable on a
// host: see firmware/test/test_lx_servo.c.
//
// Wire format (half-duplex TTL UART, 115200 8N1):
//   0x55 0x55  ID  LEN  CMD  PRM1..PRMn  CHK
//   LEN = n + 3            (counts LEN, CMD, CHK)
//   CHK = (uint8_t)~(ID + LEN + CMD + PRM1 + … + PRMn)
//
// Commands used here:
//   1  SERVO_MOVE_TIME_WRITE   params: posL posH timeL timeH   (pos 0..1000 = 0..240°)
//   26 SERVO_TEMP_READ         response param: temp_C (1 byte)
//   27 SERVO_VIN_READ          response params: mV (uint16 LE)
//   28 SERVO_POS_READ          response params: pos (int16 LE)
//   31 SERVO_OR_MOTOR_MODE_WRITE? (not used)
//   31 SERVO_LOAD_OR_UNLOAD_WRITE  param: 1 = torque on, 0 = free
#ifndef GRIPPER_LX_SERVO_H
#define GRIPPER_LX_SERVO_H

#include <stdint.h>

#define LX_HEADER            0x55
#define LX_BROADCAST_ID      254

#define LX_MOVE_TIME_WRITE   1
#define LX_TEMP_READ         26
#define LX_VIN_READ          27
#define LX_POS_READ          28
#define LX_LOAD_WRITE        31

// Position scale: 1000 units span 240 degrees.
#define LX_UNITS_PER_DEG     (1000.0f / 240.0f)
#define LX_POS_MIN_UNIT      0
#define LX_POS_MAX_UNIT      1000

static inline uint8_t lx_checksum(uint8_t id, uint8_t len, uint8_t cmd,
                                  const uint8_t *params, uint8_t nparams) {
  uint16_t s = (uint16_t)id + len + cmd;
  for (uint8_t i = 0; i < nparams; i++) s += params[i];
  return (uint8_t)(~s);
}

// Assemble a generic packet into buf. Returns total byte count (6 + nparams).
static inline uint8_t lx_build(uint8_t *buf, uint8_t id, uint8_t cmd,
                               const uint8_t *params, uint8_t nparams) {
  uint8_t len = (uint8_t)(nparams + 3);
  buf[0] = LX_HEADER;
  buf[1] = LX_HEADER;
  buf[2] = id;
  buf[3] = len;
  buf[4] = cmd;
  for (uint8_t i = 0; i < nparams; i++) buf[5 + i] = params[i];
  buf[5 + nparams] = lx_checksum(id, len, cmd, params, nparams);
  return (uint8_t)(6 + nparams);
}

static inline uint8_t lx_build_move(uint8_t *buf, uint8_t id,
                                    uint16_t pos, uint16_t time_ms) {
  if (pos > LX_POS_MAX_UNIT) pos = LX_POS_MAX_UNIT;
  uint8_t p[4] = {(uint8_t)(pos & 0xFF), (uint8_t)(pos >> 8),
                  (uint8_t)(time_ms & 0xFF), (uint8_t)(time_ms >> 8)};
  return lx_build(buf, id, LX_MOVE_TIME_WRITE, p, 4);
}

static inline uint8_t lx_build_load(uint8_t *buf, uint8_t id, uint8_t on) {
  uint8_t p = on ? 1 : 0;
  return lx_build(buf, id, LX_LOAD_WRITE, &p, 1);
}

static inline uint8_t lx_build_read(uint8_t *buf, uint8_t id, uint8_t cmd) {
  return lx_build(buf, id, cmd, 0, 0);  // read requests carry no params
}

// Validate a fully-received frame's header + checksum. `len` is the total bytes.
static inline int lx_frame_valid(const uint8_t *buf, uint8_t len) {
  if (len < 6 || buf[0] != LX_HEADER || buf[1] != LX_HEADER) return 0;
  uint8_t L = buf[3];
  if ((uint8_t)(L + 3) != len) return 0;          // LEN must match byte count
  uint8_t nparams = (uint8_t)(L - 3);
  uint8_t chk = lx_checksum(buf[2], buf[3], buf[4], &buf[5], nparams);
  return chk == buf[len - 1];
}

// Parse a response carrying an int16 LE value (POS_READ, VIN_READ). Returns 1 on
// success and writes out_id / out_val.
static inline int lx_parse_value16(const uint8_t *buf, uint8_t len, uint8_t expect_cmd,
                                   uint8_t *out_id, int16_t *out_val) {
  if (!lx_frame_valid(buf, len) || buf[4] != expect_cmd || len < 8) return 0;
  if (out_id) *out_id = buf[2];
  if (out_val) *out_val = (int16_t)((uint16_t)buf[5] | ((uint16_t)buf[6] << 8));
  return 1;
}

// Parse a response carrying a single byte value (TEMP_READ).
static inline int lx_parse_value8(const uint8_t *buf, uint8_t len, uint8_t expect_cmd,
                                  uint8_t *out_id, uint8_t *out_val) {
  if (!lx_frame_valid(buf, len) || buf[4] != expect_cmd || len < 7) return 0;
  if (out_id) *out_id = buf[2];
  if (out_val) *out_val = buf[5];
  return 1;
}

#endif  // GRIPPER_LX_SERVO_H
