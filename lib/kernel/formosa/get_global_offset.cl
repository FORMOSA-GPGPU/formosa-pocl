#include "wg_info.h"

size_t _CL_OVERLOADABLE get_global_offset(uint dim) {
  if (dim > 2)
    return 0;
  else
    return get_wg_info()->global_offset[dim];
}
