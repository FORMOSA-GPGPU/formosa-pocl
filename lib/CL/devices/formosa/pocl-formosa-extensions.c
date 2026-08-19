#include <stddef.h>

#include "config.h"

#ifdef ENABLE_FORMOSA_WORKGRAPH
extern void *pocl_work_graph_get_extension_function_address(
    const char *func_name);
#endif
extern void *pocl_formosa_stack_remap_get_extension_function_address(
    const char *func_name);

void *pocl_formosa_get_extension_function_address(const char *func_name) {
  void *function =
      pocl_formosa_stack_remap_get_extension_function_address(func_name);
  if (function != NULL) return function;

#ifdef ENABLE_FORMOSA_WORKGRAPH
  function = pocl_work_graph_get_extension_function_address(func_name);
  if (function != NULL) return function;
#endif

  return NULL;
}
