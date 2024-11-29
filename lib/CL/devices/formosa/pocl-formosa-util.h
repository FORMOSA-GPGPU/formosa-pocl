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
  int client_fd;
  uint64_t buf_address;
  uint64_t buf_size;
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

int fsa_check_occupancy(uint32_t group_size, uint32_t *max_local_mem);

int fsa_copy_to_dev(formosa_buffer_data_t *buffer_data, const void *host_ptr,
                    uint64_t dst_offset, size_t size);

int fsa_copy_from_dev(formosa_buffer_data_t *buffer_data, void *host_ptr,
                      uint64_t src_offset, size_t size);

int fsa_upload_kernel_file(const char *filename,
                           pocl_formosa_data_t *formosa_data);

int fsa_write_csr(pocl_formosa_data_t *dd, uint64_t addr, uint64_t value);

int fsa_read_csr(pocl_formosa_data_t *dd, uint64_t addr, uint64_t *value);

int fsa_wait_ack(pocl_formosa_data_t *dd);

#define FSA_WRITE_GROUPED_CSR(dd, addr, value)             \
  do {                                                     \
    int err = fsa_write_csr((dd), (addr##_X), (value[0])); \
    err |= fsa_write_csr((dd), (addr##_Y), (value[1]));    \
    err |= fsa_write_csr((dd), (addr##_Z), (value[2]));    \
    if (err == -1) {                                       \
      POCL_ABORT("FSA_WRITE_GROUPED_CSR");                 \
    }                                                      \
  } while (0);

#ifdef __cplusplus
}
#endif

#endif
