#include "wg_info.h"

void* fsa_local_alloc(void) {
  return (void*)get_wg_info()->local_memory_base;
}
