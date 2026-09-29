#include "wg_info.h"

uint _CL_OVERLOADABLE get_work_dim(void) {
  return get_wg_info()->dim;
}
