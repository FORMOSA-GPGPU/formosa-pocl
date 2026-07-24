#ifndef CL_FORMOSA_STACK_REMAP_H
#define CL_FORMOSA_STACK_REMAP_H

#include <CL/cl.h>

#ifdef __cplusplus
extern "C" {
#endif

#define CL_FORMOSA_STACK_REMAP_EXTENSION_NAME "cl_formosa_stack_remap"

/*
 * Controls designation of the automatically allocated stack regions of
 * subsequent enqueues of kernel for FORMOSA stack remapping. Designation is
 * enabled by default; CL_FALSE removes it for subsequent enqueues, while
 * CL_TRUE restores it.
 */
extern CL_API_ENTRY cl_int CL_API_CALL
clSetKernelStackRemapFORMOSA(cl_kernel kernel, cl_bool designate);

typedef cl_int(CL_API_CALL *clSetKernelStackRemapFORMOSA_fn)(cl_kernel kernel,
                                                             cl_bool designate);

#ifdef __cplusplus
}
#endif

#endif /* CL_FORMOSA_STACK_REMAP_H */
