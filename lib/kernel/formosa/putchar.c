#include <stdint.h>

#define HART_CONSOLE_BASE 0xd000

void _putchar(char character) {
  uint64_t warp_id, num_lanes, lane_id;
  __asm__ volatile("csrr %0, xwid" : "=r"(warp_id));
  __asm__ volatile("csrr %0, xlanes" : "=r"(num_lanes));
  __asm__ volatile("csrr %0, xlaneid" : "=r"(lane_id));
  uint64_t thread_offset = warp_id * num_lanes + lane_id;
  volatile char *output = (char *)(HART_CONSOLE_BASE + thread_offset);
  *output = character;
}
