#ifndef FORMOSA_WG_INFO_H_
#define FORMOSA_WG_INFO_H_

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
};

static inline struct WGInfo *get_wg_info() {
  struct WGInfo *info;
  __asm__ volatile("csrr %0, mscratch" : "=r"(info));
  return info;
}

#endif
