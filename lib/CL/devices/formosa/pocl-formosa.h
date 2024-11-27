#ifndef POCL_FORMOSA_H
#define POCL_FORMOSA_H

#include "pocl_cl.h"
#include "prototypes.inc"

typedef struct {
  int num_kernels;
  char *kernel_names;
} formosa_program_data_t;

typedef struct {
  size_t ref_count;
  int kernel_id;
} formosa_kernel_data_t;

typedef struct {
  int client_fd;
  uint64_t buf_address;
  uint64_t buf_size;
} formosa_buffer_data_t;

typedef struct {
  uint32_t kernel_id;
  uint32_t work_dim;
  uint32_t local_size[3];
  uint32_t num_groups[3];
} kernel_args_t;

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

  /* The file descriptor of the virtual device */
  int client_fd;

  /* The kernel data buffer */
  formosa_buffer_data_t *kernel_buffer;
} pocl_formosa_data_t;

typedef struct _pocl_basic_usm_allocation_t {
  void *ptr;
  size_t size;
  cl_mem_alloc_flags_intel flags;
  unsigned alloc_type;

  struct _pocl_basic_usm_allocation_t *next, *prev;
} pocl_formosa_usm_allocation_t;

// align the offset to the alignment
uint32_t align_offset(uint32_t offset, uint32_t alignment) {
  return (offset + alignment - 1) & ~(alignment - 1);
}

// check if the kernel can be scheduled on the device
int check_occupancy(uint32_t group_size, uint32_t *max_local_mem);

int fsa_copy_to_dev(formosa_buffer_data_t *buffer_data, const void *host_ptr,
                    uint64_t dst_offset, size_t size);

int fsa_copy_from_dev(formosa_buffer_data_t *buffer_data, void *host_ptr,
                      uint64_t src_offset, size_t size);

int fsa_write_csr(pocl_formosa_data_t *dd, uint64_t addr, uint64_t value);

int fsa_read_csr(pocl_formosa_data_t *dd, uint64_t addr, uint64_t *value);

int fsa_upload_kernel_file(const char *filename,
                           formosa_buffer_data_t *buffer_data);

int fsa_upload_kernel_bytes(const void *data, uint64_t size,
                            formosa_buffer_data_t *buffer_data);

int fsa_ready_wait(pocl_formosa_data_t *dd);

GEN_PROTOTYPES(formosa)

#endif /* POCL_FORMOSA_H */
