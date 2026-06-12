// AddressSanitizer + UBSan fuzz of the LX byte parser. Throws millions of
// random-length, random-byte (and near-valid-then-mutated) buffers at the parse
// helpers; any out-of-bounds read or UB aborts non-zero with a sanitizer report.
//   cc -std=c11 -O1 -fsanitize=address,undefined -fno-sanitize-recover=all \
//      -Wall -Wextra -o /tmp/fuzz_lx firmware/test/fuzz_lx.c && /tmp/fuzz_lx
#include <stdio.h>
#include <stdlib.h>
#include "../esp32_servo/lx_servo.h"

int main(int argc, char **argv) {
  unsigned long M = (argc > 1) ? strtoul(argv[1], 0, 10) : 3000000UL;
  srand(0xBADC0DE);
  unsigned long accepted = 0;
  uint8_t buf[64];
  for (unsigned long i = 0; i < M; i++) {
    int len = rand() % (int)sizeof(buf);
    for (int j = 0; j < len; j++) buf[j] = (uint8_t)(rand() & 0xFF);
    // 1/3 of the time, craft a length-consistent header then leave random body
    if (i % 3 == 0 && len >= 6) {
      buf[0] = LX_HEADER; buf[1] = LX_HEADER; buf[3] = (uint8_t)(len - 3);
    }
    uint8_t id, v8;
    int16_t v16;
    int a = lx_frame_valid(buf, (uint8_t)len);
    int b = lx_parse_value16(buf, (uint8_t)len, LX_POS_READ, &id, &v16);
    int c = lx_parse_value8(buf, (uint8_t)len, LX_TEMP_READ, &id, &v8);
    accepted += (unsigned)(a || b || c);
  }
  // also fuzz round-trips: build, then mutate one byte, re-validate
  for (unsigned long i = 0; i < M / 4; i++) {
    uint8_t pkt[16];
    uint8_t n = lx_build_move(pkt, (uint8_t)(rand() & 0xFF),
                              (uint16_t)(rand() % 1001), (uint16_t)(rand() % 30000));
    if (n > 0) {
      pkt[rand() % n] ^= (uint8_t)(1 << (rand() % 8));  // flip a bit
      (void)lx_frame_valid(pkt, n);
    }
  }
  printf("lx fuzz: %lu iters, %lu accepted, no ASan/UBSan faults\n", M, accepted);
  return 0;
}
