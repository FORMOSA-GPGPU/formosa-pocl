#include "pocl-formosa.h"

#include "common.h"
#include "common_driver.h"
#include "pocl_util.h"

typedef struct {
  /* List of commands ready to be executed */
  _cl_command_node *ready_list;

  /* List of commands not yet ready to be executed */
  _cl_command_node *command_list;

  /* Lock for command list related operations */
  pocl_lock_t cq_lock;

  /* The number of contexts that are currently using this device */
  size_t context_ref_count;
} pocl_formosa_data_t;

typedef struct _pocl_basic_usm_allocation_t {
  void *ptr;
  size_t size;
  cl_mem_alloc_flags_intel flags;
  unsigned alloc_type;

  struct _pocl_basic_usm_allocation_t *next, *prev;
} pocl_formosa_usm_allocation_t;

typedef struct {
  int num_kernels;
  char *kernel_names;
} formosa_program_data_t;

typedef struct {
  size_t refcount;
  int kernel_id;
} formosa_kernel_data_t;

void pocl_formosa_init_device_ops(struct pocl_device_ops *ops) {
  ops->device_name = "formosa";
  ops->build_hash = pocl_formosa_build_hash;
  ops->probe = pocl_formosa_probe;
  ops->uninit = pocl_formosa_uninit;
  ops->init = pocl_formosa_init;
  ops->reinit = pocl_formosa_reinit;

  ops->init_context = pocl_formosa_init_context;
  ops->free_context = pocl_formosa_free_context;

  ops->run = pocl_formosa_run;
  ops->run_native = NULL;

  ops->alloc_mem_obj = pocl_formosa_alloc_mem_obj;
  ops->free = pocl_formosa_free;

  ops->init_build = pocl_formosa_init_build;
  ops->build_source = pocl_driver_build_source;
  ops->link_program = pocl_driver_link_program;
  ops->build_binary = pocl_driver_build_binary;
  ops->free_program = pocl_driver_free_program;
  ops->setup_metadata = pocl_driver_setup_metadata;
  ops->supports_binary = pocl_driver_supports_binary;
  ops->build_poclbinary = pocl_driver_build_poclbinary;
  ops->build_builtin = pocl_driver_build_opencl_builtins;

  ops->post_build_program = pocl_formosa_post_build_program;
  ops->free_program = pocl_formosa_free_program;

  ops->create_kernel = pocl_formosa_create_kernel;
  ops->free_kernel = pocl_formosa_free_kernel;

  ops->submit = pocl_formosa_submit;
  ops->join = pocl_formosa_join;
  ops->flush = pocl_formosa_flush;
  ops->notify = pocl_formosa_notify;
  ops->broadcast = pocl_broadcast;

  ops->read = pocl_formosa_read;
  ops->write = pocl_formosa_write;
  ops->copy = pocl_formosa_copy;

  ops->get_mapping_ptr = pocl_driver_get_mapping_ptr;
  ops->free_mapping_ptr = pocl_driver_free_mapping_ptr;
}

/**************************
 * Probe & Initialization *
 **************************/

unsigned int pocl_formosa_probe(struct pocl_device_ops *ops) {
  return strcmp(ops->device_name, "formosa") == 0;
}

char *pocl_formosa_build_hash(cl_device_id device) {
  char *res = (char *)calloc(1000, sizeof(char));
  snprintf(res, 1000, "formosa-riscv64-unknown-unknwon-elf");
  return res;
}

cl_int pocl_formosa_init(unsigned j, cl_device_id device,
                         const char *parameters) {}

cl_int pocl_formosa_uninit(unsigned j, cl_device_id device) {}

cl_int pocl_formosa_reinit(unsigned j, cl_device_id device,
                           const char *parameters) {}

void pocl_formosa_run(void *data, _cl_command_node *cmd) {}

/**************************
 * Program                *
 **************************/

char *pocl_formosa_init_build(void *data) {}

int pocl_formosa_post_build_program(cl_program program, cl_uint device_i) {}

int pocl_formosa_free_program(cl_device_id device, cl_program program,
                              unsigned program_device_i) {}

/**************************
 * Kernel                 *
 **************************/

cl_int pocl_formosa_create_kernel(cl_device_id device, cl_program program,
                                  cl_kernel kernel, unsigned program_device_i) {
  int result = CL_SUCCESS;
  pocl_kernel_metadata_t *meta = kernel->meta;
  assert(meta->data != NULL);
  // device-specific kernel metadata
  formosa_kernel_data_t *kdata =
      (formosa_kernel_data_t *)meta->data[program_device_i];
  if (kdata != NULL) {
    ++kdata->refcount;
    return CL_SUCCESS;
  }

  do {
    // device-specific program data
    formosa_program_data_t *pdata =
        (formosa_program_data_t *)program->data[program_device_i];
    assert(pdata != NULL);

    const char *current = pdata->kernel_names;
    int i = 0;
    int found = 0;
    for (; i < pdata->num_kernels; ++i) {
      if (strcmp(current, kernel->name) == 0) {
        found = 1;
        break;
      }
      current += strlen(current) + 1;
    }
    assert(found);
    kdata = (void *)calloc(1, sizeof(formosa_kernel_data_t));
    kdata->kernel_id = i;
    ++kdata->refcount;

  } while (0);

  meta->data[program_device_i] = kdata;

  return result;
}

cl_int pocl_formosa_free_kernel(cl_device_id device, cl_program program,
                                cl_kernel kernel, unsigned program_device_i) {
  pocl_kernel_metadata_t *meta = kernel->meta;
  assert(meta->data != NULL);
  formosa_kernel_data_t *kdata =
      (formosa_kernel_data_t *)meta->data[program_device_i];
  if (kdata == NULL) return CL_SUCCESS;

  --kdata->refcount;
  if (kdata->refcount == 0) {
    POCL_MEM_FREE(kdata);
    meta->data[program_device_i] = NULL;
  }

  return CL_SUCCESS;
}

/**************************
 * Context                *
 **************************/

cl_int pocl_formosa_init_context(cl_device_id device, cl_context context) {
  pocl_formosa_data_t *dd = (pocl_formosa_data_t *)device->data;
  if (NULL == dd) return CL_SUCCESS;

  dd->context_ref_count++;

  return CL_SUCCESS;
}

cl_int pocl_formosa_free_context(cl_device_id device, cl_context context) {
  pocl_formosa_data_t *dd = (pocl_formosa_data_t *)device->data;
  if (NULL == dd) return CL_SUCCESS;

  dd->context_ref_count--;
  if (dd->context_ref_count == 0) {
    pocl_formosa_uninit(0, device);
  }

  return CL_SUCCESS;
}

/**************************
 * Memory Allocation      *
 **************************/

cl_int pocl_formosa_alloc_mem_obj(cl_device_id device, cl_mem mem_obj,
                                  void *host_ptr) {}

void pocl_formosa_free(cl_device_id device, cl_mem mem_obj) {}

/**************************
 * Event Handling         *
 **************************/

static void formosa_command_scheduler(pocl_formosa_data_t *d) {
  _cl_command_node *node;

  /* execute commands from ready list */
  while ((node = d->ready_list)) {
    assert(pocl_command_is_ready(node->sync.event.event));
    assert(node->sync.event.event->status == CL_SUBMITTED);
    CDL_DELETE(d->ready_list, node);
    POCL_UNLOCK(d->cq_lock);
    pocl_exec_command(node);
    POCL_LOCK(d->cq_lock);
  }
}

void pocl_formosa_submit(_cl_command_node *node, cl_command_queue cq) {
  pocl_formosa_data_t *d = node->device->data;

  if (node != NULL && node->type == CL_COMMAND_NDRANGE_KERNEL) {
    cl_kernel kernel = node->command.run.kernel;
    cl_program program = kernel->program;
    if (!program->builtin_kernel_attributes) {
      node->command.run.device_data =
          pocl_check_kernel_dlhandle_cache(node, CL_TRUE, CL_TRUE);
    }
  }

  node->ready = 1;
  POCL_LOCK(d->cq_lock);
  pocl_command_push(node, &d->ready_list, &d->command_list);

  POCL_UNLOCK_OBJ(node->sync.event.event);
  formosa_command_scheduler(d);
  POCL_UNLOCK(d->cq_lock);
}

void pocl_formosa_join(cl_device_id device, cl_command_queue cq) {
  pocl_formosa_data_t *d = (pocl_formosa_data_t *)device->data;

  POCL_LOCK(d->cq_lock);
  formosa_command_scheduler(d);
  POCL_UNLOCK(d->cq_lock);
}

void pocl_formosa_flush(cl_device_id device, cl_command_queue cq) {
  pocl_formosa_data_t *d = (pocl_formosa_data_t *)device->data;

  POCL_LOCK(d->cq_lock);
  formosa_command_scheduler(d);
  POCL_UNLOCK(d->cq_lock);
}

void pocl_formosa_notify(cl_device_id device, cl_event event,
                         cl_event finished) {
  pocl_formosa_data_t *d = (pocl_formosa_data_t *)device->data;
  _cl_command_node *volatile node = event->command;

  if (finished->status < CL_COMPLETE) {
    pocl_update_event_failed(event);
    return;
  }

  if (!node->ready) return;

  if (pocl_command_is_ready(event)) {
    if (event->status == CL_QUEUED) {
      pocl_update_event_submitted(event);
      POCL_LOCK(d->cq_lock);
      CDL_DELETE(d->command_list, node);
      CDL_PREPEND(d->ready_list, node);
      POCL_UNLOCK_OBJ(event);
      formosa_command_scheduler(d);
      POCL_LOCK_OBJ(event);
      POCL_UNLOCK(d->cq_lock);
    }
  }
}

/************************
 * Command Execution    *
 * **********************/

void pocl_formosa_read(void *data, void *__restrict__ host_ptr,
                       pocl_mem_identifier *src_mem_id, cl_mem src_buf,
                       size_t offset, size_t size) {}

void pocl_formosa_write(void *data, const void *__restrict__ host_ptr,
                        pocl_mem_identifier *dst_mem_id, cl_mem dst_buf,
                        size_t offset, size_t size) {}

void pocl_formosa_copy(void *data, pocl_mem_identifier *dst_mem_id,
                       cl_mem dst_buf, pocl_mem_identifier *src_mem_id,
                       cl_mem src_buf, size_t dst_offset, size_t src_offset,
                       size_t size) {}
