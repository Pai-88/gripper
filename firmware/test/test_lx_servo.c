// Host test for lx_servo.h — verifies the LX packet byte layout, checksum, and
// response parsing against the documented LewanSoul reference vectors.
//
//   cc -std=c11 -Wall -Wextra -o /tmp/test_lx firmware/test/test_lx_servo.c && /tmp/test_lx
#include <stdio.h>
#include <string.h>
#include "../esp32_servo/lx_servo.h"

static int failures = 0;

static void check_bytes(const char *name, const uint8_t *got, uint8_t got_len,
                        const uint8_t *want, uint8_t want_len) {
  if (got_len != want_len || memcmp(got, want, want_len) != 0) {
    printf("FAIL %s\n  got :", name);
    for (uint8_t i = 0; i < got_len; i++) printf(" %02X", got[i]);
    printf("\n  want:");
    for (uint8_t i = 0; i < want_len; i++) printf(" %02X", want[i]);
    printf("\n");
    failures++;
  } else {
    printf("ok   %s\n", name);
  }
}

static void check(const char *name, int cond) {
  printf("%s %s\n", cond ? "ok  " : "FAIL", name);
  if (!cond) failures++;
}

int main(void) {
  uint8_t buf[16];
  uint8_t n;

  // 1) MOVE id=1 -> pos 1000 (0x03E8) over 1000 ms (0x03E8). Reference packet.
  n = lx_build_move(buf, 1, 1000, 1000);
  const uint8_t want_move[] = {0x55,0x55,0x01,0x07,0x01,0xE8,0x03,0xE8,0x03,0x20};
  check_bytes("move(1,1000,1000)", buf, n, want_move, sizeof want_move);

  // 2) READ position request id=1, cmd 28 (0x1C).
  n = lx_build_read(buf, 1, LX_POS_READ);
  const uint8_t want_read[] = {0x55,0x55,0x01,0x03,0x1C,0xDF};
  check_bytes("read_pos_req(1)", buf, n, want_read, sizeof want_read);

  // 3) LOAD (torque on) id=1, cmd 31 (0x1F), param 1.
  n = lx_build_load(buf, 1, 1);
  const uint8_t want_load[] = {0x55,0x55,0x01,0x04,0x1F,0x01,0xDA};
  check_bytes("load_on(1)", buf, n, want_load, sizeof want_load);

  // 4) Built frames must self-validate.
  n = lx_build_move(buf, 3, 750, 20);
  check("frame_valid(move)", lx_frame_valid(buf, n));
  buf[n - 1] ^= 0xFF;  // corrupt checksum
  check("frame_invalid(bad crc)", !lx_frame_valid(buf, n));

  // 5) Parse a POS_READ response (pos = 500 = 0x01F4).
  uint8_t pos_resp[] = {0x55,0x55,0x01,0x05,0x1C,0xF4,0x01,0xE8};
  uint8_t id = 0; int16_t val = 0;
  check("parse pos frame valid", lx_frame_valid(pos_resp, sizeof pos_resp));
  check("parse pos value", lx_parse_value16(pos_resp, sizeof pos_resp,
                                             LX_POS_READ, &id, &val)
                           && id == 1 && val == 500);

  // 6) Parse a VIN_READ response (7400 mV = 0x1CE8).
  uint8_t vin_resp[] = {0x55,0x55,0x01,0x05,0x1B,0xE8,0x1C,0xDA};
  check("parse vin value", lx_parse_value16(vin_resp, sizeof vin_resp,
                                            LX_VIN_READ, &id, &val)
                           && val == 7400);

  // 7) Parse a TEMP_READ response (53 C = 0x35).
  uint8_t temp_resp[] = {0x55,0x55,0x01,0x04,0x1A,0x35,0xAB};
  uint8_t t = 0;
  check("parse temp value", lx_parse_value8(temp_resp, sizeof temp_resp,
                                            LX_TEMP_READ, &id, &t) && t == 53);

  // 8) Wrong-command parse is rejected.
  check("reject wrong cmd", !lx_parse_value16(pos_resp, sizeof pos_resp,
                                              LX_VIN_READ, &id, &val));

  printf("\n%s (%d failure%s)\n", failures ? "FAILED" : "PASSED",
         failures, failures == 1 ? "" : "s");
  return failures ? 1 : 0;
}
