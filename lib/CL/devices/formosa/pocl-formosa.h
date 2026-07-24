#ifndef POCL_FORMOSA_H
#define POCL_FORMOSA_H

#include "pocl_cl.h"
#include "prototypes.inc"

GEN_PROTOTYPES(formosa)

cl_int pocl_formosa_set_kernel_stack_remap(cl_device_id device,
                                           unsigned program_device_i,
                                           cl_kernel kernel, cl_bool designate);
cl_bool pocl_formosa_kernel_stack_remap_enabled(cl_kernel kernel,
                                                cl_device_id device);

#endif /* POCL_FORMOSA_H */
