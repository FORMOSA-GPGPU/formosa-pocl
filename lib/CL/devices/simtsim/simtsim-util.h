#ifndef SIMTSIM_UTIL_H
#define SIMTSIM_UTIL_H

#include "pocl.h"
#include "pocl_threads_c.h"

#ifdef __cplusplus
extern "C" {
#endif

#include "simtsim.h"
#include <stddef.h>
#include <stdint.h>

typedef struct {
  uint64_t addr;
  uint64_t size;
} simtsim_buffer_data_t;

typedef struct {
  int num_kernels;
  char *names;
  SimtsimProgram *sim_program;
} simtsim_program_data_t;

typedef struct {
  size_t ref_count;
  int id;
} simtsim_kernel_data_t;

typedef struct {
  _cl_command_node *ready_list;
  _cl_command_node *command_list;
  pocl_lock_t cq_lock;
  pocl_lock_t compile_lock;
  size_t ref_count;
  simtsim_buffer_data_t *kernel_buffer;
  SimtsimSystem *sim;
} pocl_simtsim_data_t;

int pocl_simtsim_get_elf_name(cl_program program, cl_uint device_i,
                              char *elf_name);
int pocl_simtsim_compile_program(char **kernel_names, int *num_kernels,
                                 char *str_program_simtsim_bin,
                                 char *compiler_options, void *llvm_module);
int pocl_simtsim_check_kernel_valid(uint32_t group_size);

#ifdef __cplusplus
}
#endif

#endif