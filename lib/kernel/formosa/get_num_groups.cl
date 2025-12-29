#include "wg_info.h"

size_t _CL_OVERLOADABLE get_num_groups(uint dim) {
  if (dim > 2)
    return 1;
  else
    return get_wg_info()->num_groups[dim];
}
