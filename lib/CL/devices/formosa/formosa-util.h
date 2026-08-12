#ifndef FORMOSA_UTIL_H
#define FORMOSA_UTIL_H

#include <formosa-hal/formosa-hal.h>

#include "pocl.h"

#ifdef __cplusplus
extern "C" {
#endif

#include <stddef.h>
#include <stdint.h>

int pocl_fsa_check_occupancy(uint32_t group_size, uint64_t local_mem_per_group,
                             uint64_t *max_local_mem);

int pocl_fsa_get_elf_name(cl_program program, cl_uint device_i, char *elf_name);

int pocl_fsa_upload_kernel(const char *elf_file, uint64_t *kernel_dev_addr);

cl_int pocl_fsa_wait_completion(FsaCompletionToken token,
                                uintptr_t device_kernel_status_addr);

int pocl_fsa_compile_program(char **kernel_names, int *num_kernels,
                             char *str_program_fsa_bin, char *compiler_options,
                             void *llvm_module);

#define FSA_TASK_DISPATCHER_BASE 0x1000
#define FSA_GLOBAL_MEM_BASE                      \
  ((0x80000000) +                                \
   ((fsa_num_cores()) * (fsa_warps_per_core()) * \
    (fsa_threads_per_warp() * (fsa_stack_size_per_thread()))

#ifdef __cplusplus
}
#endif

#endif  // FORMOSA_UTIL_H
