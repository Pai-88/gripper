// limits.h — GENERATED from config/robot.yaml by tools/gen_firmware_limits.py.
// DO NOT EDIT BY HAND. Regenerate after changing robot.yaml.
// joint order: base, shoulder, elbow, wrist
#ifndef GRIPPER_LIMITS_H
#define GRIPPER_LIMITS_H

#include "protocol.h"

#define MCU_WATCHDOG_MS 200
#define SERVO_WRITE_HZ  50
#define MOVE_TIME_MS    20
#define GRIP_SERVO_ID   1
#define GRIP_MAX_STEP   60
#define GRIP_POS_CLOSED 291
#define GRIP_POS_OPEN   709

static const uint8_t SERVO_ID[JOINT_COUNT] = { 6, 5, 4, 3 };
static const int32_t JOINT_SPAN_CD[JOINT_COUNT] = { 18000, 9000, 12000, 18000 };
static const int32_t HOME_CD[JOINT_COUNT] = { 9000, 2000, 6000, 9000 };
static const int32_t JOINT_MAX_STEP_CD[JOINT_COUNT] = { 240, 180, 240, 300 };
static const float JOINT_MIN_DEG[JOINT_COUNT] = { -90.0f, 0.0f, -120.0f, -90.0f };
static const int32_t SERVO_POS_MIN[JOINT_COUNT] = { 125, 312, 250, 125 };
static const int32_t SERVO_POS_MAX[JOINT_COUNT] = { 875, 688, 750, 875 };

#endif  // GRIPPER_LIMITS_H
