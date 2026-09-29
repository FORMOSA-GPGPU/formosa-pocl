#include "wg_info.h"

size_t _CL_OVERLOADABLE get_group_id(uint dim) {
  if (dim > 2)
    return 1;
  else
    return get_wg_info()->wg_id[dim];
}
