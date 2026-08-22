#include <stddef.h>

#include "config.h"

extern void *pocl_formosa_stack_remap_get_extension_function_address(
    const char *func_name);

void *pocl_formosa_get_extension_function_address(const char *func_name) {
  return pocl_formosa_stack_remap_get_extension_function_address(func_name);
}
