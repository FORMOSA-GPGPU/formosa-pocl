#ifndef POCL_FORMOSA_INTERNAL_H
#define POCL_FORMOSA_INTERNAL_H

#include "formosa-memory.h"
#include "pocl-formosa.h"
#include "pocl_threads.h"

typedef struct {
  int num_kernels;
  char *kernel_names;
} formosa_program_data_t;

typedef struct {
  size_t ref_count;
  int kernel_id;
} formosa_kernel_data_t;

typedef struct {
  cl_bool stack_remap;
} formosa_kernel_instance_data_t;

typedef struct {
  uint64_t printf_buffer;
  uint64_t printf_buffer_position;
  uint32_t printf_buffer_capacity;
  uint32_t reserved;
} formosa_printf_launch_meta_t;

typedef struct formosa_pending_copy {
  _cl_command_node *node;
  FsaMemoryCopyCompletion completion;
  struct formosa_pending_copy *next;
} formosa_pending_copy_t;

/* PoCL device-specific state.  This is private to the Formosa backend and is
 * intentionally not part of the generic Formosa utility interface. */
typedef struct {
  /* List of commands ready to be executed */
  _cl_command_node *ready_list;

  /* List of commands not yet ready to be executed */
  _cl_command_node *command_list;

  /* Lock for command list related operations */
  pocl_lock_t cq_lock;

  /* Lock for compile operations */
  pocl_lock_t compile_lock;

  /* Completion adapter for asynchronous H2D/D2H/D2D commands. */
  pocl_lock_t copy_lock;
  pocl_cond_t copy_cond;
  pocl_thread_t copy_thread;
  cl_bool copy_thread_stop;
  cl_bool copy_thread_started;
  formosa_pending_copy_t *copy_pending;
  formosa_pending_copy_t *copy_pending_tail;

  /* The number of contexts that are currently using this device */
  size_t context_ref_count;

  /* The kernel data buffer */
  formosa_buffer_data_t *kernel_buffer;

  /* Message ID */
  uint64_t msg_id;
} pocl_formosa_data_t;

#endif  // POCL_FORMOSA_INTERNAL_H
