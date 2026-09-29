#include "wg_info.h"

size_t _CL_OVERLOADABLE get_local_size(uint dim) {
  if (dim > 2)
    return 1;
  else
    return get_wg_info()->local_size[dim];
}
