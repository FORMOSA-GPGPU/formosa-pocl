#ifndef POCL_FORMOSA_UTIL_H
#define POCL_FORMOSA_UTIL_H

#include "pocl.h"
#include "pocl_threads.h"

#ifdef __cplusplus
extern "C" {
#endif

#include <stddef.h>
#include <stdint.h>

typedef struct {
  int num_kernels;
  char *kernel_names;
} formosa_program_data_t;

typedef struct {
  size_t ref_count;
  int kernel_id;
} formosa_kernel_data_t;

typedef struct {
  uint64_t buf_address;
  uint64_t buf_size;
  uint64_t msg_id;
} formosa_buffer_data_t;

// device specific data
typedef struct {
  /* List of commands ready to be executed */
  _cl_command_node *ready_list;

  /* List of commands not yet ready to be executed */
  _cl_command_node *command_list;

  /* Lock for command list related operations */
  pocl_lock_t cq_lock;

  /* Lock for compile operations */
  pocl_lock_t compile_lock;

  /* The number of contexts that are currently using this device */
  size_t context_ref_count;

  /* The kernel data buffer */
  formosa_buffer_data_t *kernel_buffer;

  /* Message ID */
  uint64_t msg_id;
} pocl_formosa_data_t;

typedef struct _pocl_basic_usm_allocation_t {
  void *ptr;
  size_t size;
  cl_mem_alloc_flags_intel flags;
  unsigned alloc_type;

  struct _pocl_basic_usm_allocation_t *next, *prev;
} pocl_formosa_usm_allocation_t;

int fsa_check_occupancy(uint32_t group_size, uint32_t *max_local_mem);

int fsa_get_elf_name(cl_program program, cl_uint device_i, char *elf_name);

int fsa_upload_kernel_sections(const char *filename,
                               pocl_formosa_data_t *formosa_data);

int fsa_upload_kernel(const char *elf_file, pocl_formosa_data_t *dd, uint64_t *kernel_dev_addr, uint64_t *kernel_base);
void fsa_int_handler(int sig);

int fsa_wait_ack(pocl_formosa_data_t *dd);

int fsa_compile_program(char **kernel_names, int *num_kernels,
                        char *str_program_fsa_bin, char *compiler_options,
                        void *llvm_module);

uint64_t fsa_get_symbol_pc(const char *elf_path, const char *symbol_name);
uint64_t fsa_get_symbol_offset(const char *elf_path, const char *symbol_name);
#define FSA_TASK_DISPATCHER_BASE 0x1000
#define FSA_GLOBAL_MEM_BASE 0x40000000

#ifdef __cplusplus
}
#endif

#endif
