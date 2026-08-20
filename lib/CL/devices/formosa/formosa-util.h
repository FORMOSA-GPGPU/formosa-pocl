#ifndef FORMOSA_UTIL_H
#define FORMOSA_UTIL_H

#include <formosa-hal/api.h>

#include "pocl.h"

#ifdef __cplusplus
extern "C" {
#endif

#include <stddef.h>
#include <stdint.h>

int pocl_fsa_get_elf_name(cl_program program, cl_uint device_i, char *elf_name);

int pocl_fsa_upload_kernel(const char *elf_file, uint64_t *kernel_dev_addr);

typedef FsaCommandSubmitStatus (*pocl_fsa_submit_fn)(void *context,
                                                     FsaCompletionToken *token);

FsaCommandSubmitStatus pocl_fsa_submit_with_backpressure(
    pocl_fsa_submit_fn submit, void *context, FsaCompletionToken *token);

/* Wait for a terminal Completion Outcome and release its slot.  A terminal
 * command failure is returned through result, not as an interface error. */
cl_int pocl_fsa_wait_completion_result(FsaCompletionToken token,
                                       FsaCompletionResult *result);

cl_int pocl_fsa_wait_completion(FsaCompletionToken token,
                                uintptr_t device_kernel_status_addr);

int pocl_fsa_compile_program(char **kernel_names, int *num_kernels,
                             char *str_program_fsa_bin, char *compiler_options,
                             void *llvm_module);

#ifdef __cplusplus
}
#endif

#endif  // FORMOSA_UTIL_H
