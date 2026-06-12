// protocol.h — host<->MCU wire protocol, hand-mirrored from gripper/protocol.py.
// Keep BYTE-FOR-BYTE in sync; tests/test_protocol.py pins the Python side.
//
// Host -> MCU:
//   J,<base>,<shoulder>,<elbow>,<wrist>,<grip>,<seq>,<crc>\n   (centideg offset-from-min)
//   M,<mode>\n   E\n   H\n   P\n
// MCU -> Host:
//   S,<state>,<base>,<shoulder>,<elbow>,<wrist>,<grip>,<vbat_mV>,<flags>,<seq>,<crc>\n
#ifndef GRIPPER_PROTOCOL_H
#define GRIPPER_PROTOCOL_H

#include <stdint.h>
#include <stddef.h>

#define JOINT_COUNT     4
#define ANGLE_MIN_CD    0
#define ANGLE_MAX_CD    18000
#define GRIP_MIN        0
#define GRIP_MAX        1000

// modes (M frame)
#define MODE_IDLE       0
#define MODE_TELEOP     1
#define MODE_AUTO       2

// telemetry flag bits (S frame) — mirror gripper/protocol.py
#define FLAG_UNDERVOLT   (1 << 0)
#define FLAG_LIMIT_CLAMP (1 << 1)
#define FLAG_WATCHDOG    (1 << 2)
#define FLAG_OVERTEMP    (1 << 3)
#define FLAG_OVERCURRENT (1 << 4)

// state codes (S frame) — mirror enum order in gripper/state_machine.py:State
// 0 IDLE, 1 TELEOP, 2 ARM_AUTO, 3 AUTO_GRASP, 4 RETURN, 5 ESTOP

// CRC-8/MAXIM (Dallas/1-Wire): poly 0x31 reflected (0x8C), init 0x00.
// check("123456789") == 0xA1. Identical to gripper.protocol.crc8_maxim.
static inline uint8_t crc8_maxim(const uint8_t *data, size_t len) {
  uint8_t crc = 0;
  for (size_t i = 0; i < len; i++) {
    crc ^= data[i];
    for (uint8_t b = 0; b < 8; b++) {
      crc = (crc & 1) ? (uint8_t)((crc >> 1) ^ 0x8C) : (uint8_t)(crc >> 1);
    }
  }
  return crc;
}

#endif  // GRIPPER_PROTOCOL_H
