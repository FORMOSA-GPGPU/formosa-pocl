#include "CL/cl_formosa_stack_remap.h"
#include "config.h"
#include "formosa/pocl-formosa.h"
#include "pocl_util.h"

#include <string.h>

#ifdef ENABLE_FORMOSA_WORKGRAPH
extern void *pocl_work_graph_get_extension_function_address(const char *func_name);
#endif

static cl_int pocl_set_kernel_stack_remap(cl_kernel kernel, cl_bool designate) {
  cl_bool supported = CL_FALSE;

  for (cl_uint i = 0; i < kernel->program->num_devices; ++i) {
    cl_device_id device = pocl_real_dev(kernel->program->devices[i]);
    const struct pocl_stack_remap_ops *ops = NULL;
    if (device != NULL && device->ops != NULL &&
        device->ops->get_extension_ops != NULL)
      ops = (const struct pocl_stack_remap_ops *)device->ops->get_extension_ops(
          CL_FORMOSA_STACK_REMAP_EXTENSION_NAME);
    if (ops == NULL || ops->set_kernel_stack_remap == NULL) continue;

    supported = CL_TRUE;
    cl_int err =
        ops->set_kernel_stack_remap(device, i, kernel, designate);
    if (err != CL_SUCCESS) return err;
  }

  return supported ? CL_SUCCESS : CL_INVALID_OPERATION;
}

CL_API_ENTRY cl_int CL_API_CALL
POname(clSetKernelStackRemapFORMOSA)(cl_kernel kernel, cl_bool designate) {
  POCL_RETURN_ERROR_COND((!IS_CL_OBJECT_VALID(kernel)), CL_INVALID_KERNEL);
  POCL_RETURN_ERROR_ON((designate != CL_FALSE && designate != CL_TRUE),
                       CL_INVALID_VALUE,
                       "designate must be CL_FALSE or CL_TRUE\n");
  return pocl_set_kernel_stack_remap(kernel, designate);
}
POsym(clSetKernelStackRemapFORMOSA)

void *pocl_formosa_get_extension_function_address(const char *func_name) {
  if (strcmp(func_name, "clSetKernelStackRemapFORMOSA") == 0)
    return (void *)&POname(clSetKernelStackRemapFORMOSA);

#ifdef ENABLE_FORMOSA_WORKGRAPH
  return pocl_work_graph_get_extension_function_address(func_name);
#else
  return NULL;
#endif
}
