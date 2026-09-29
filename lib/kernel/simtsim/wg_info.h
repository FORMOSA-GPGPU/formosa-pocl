#ifndef FORMOSA_WG_INFO_H_
#define FORMOSA_WG_INFO_H_

#include <stddef.h>
#include <stdint.h>

// https://wiki.caslab.ee.ncku.edu.tw/doc/hardware-s6Qvv4Rq2B#h-work-group
struct WGInfo {
  uint32_t dim;
  uint32_t wg_id[3];
  uint32_t local_size[3];
  uint32_t num_groups[3];
  uint32_t global_offset[3];
  uint32_t num_threads;
  void *trampoline;
  void *kargs;
  uint64_t stack_base;  // stack base address for this work-group
  uint32_t stack_size;  // stack size for a work-item
  uint64_t local_memory_base;
  uint32_t local_memory_size;
  char *printf_buffer;
  uint32_t *printf_buffer_position;
  uint32_t printf_buffer_capacity;
  uint32_t reserved0;
};

_Static_assert(offsetof(struct WGInfo, stack_base) == 0x48,
               "WGInfo.stack_base offset mismatch");
_Static_assert(offsetof(struct WGInfo, printf_buffer) == 0x68,
               "WGInfo.printf_buffer offset mismatch");

static inline struct WGInfo *get_wg_info() {
  struct WGInfo *info;
  __asm__ volatile("csrr %0, mscratch" : "=r"(info));
  return info;
}

#endif
