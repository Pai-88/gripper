// Cross-validate the C crc8_maxim (protocol.h) against Python's, over random
// vectors piped on stdin as "HEXDATA EXPECTEDCRC" lines.
//   cc -O2 -o /tmp/crc_xcheck firmware/test/crc_xcheck.c
#include <stdio.h>
#include <string.h>
#include "../esp32_servo/protocol.h"

int main(void) {
  char hex[8200];
  long expected;
  unsigned long n = 0, bad = 0;
  while (scanf("%8199s %ld", hex, &expected) == 2) {
    size_t len = strlen(hex) / 2;
    unsigned char buf[4096];
    if (len > sizeof buf) len = sizeof buf;
    for (size_t i = 0; i < len; i++) {
      unsigned v;
      sscanf(hex + 2 * i, "%2x", &v);
      buf[i] = (unsigned char)v;
    }
    if (crc8_maxim(buf, len) != (uint8_t)expected) bad++;
    n++;
  }
  printf("crc xcheck: %lu vectors, %lu mismatches\n", n, bad);
  return bad ? 1 : 0;
}
