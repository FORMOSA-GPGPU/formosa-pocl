#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#include "CL/cl_ext.h"
#include "common.h"
#include "common_driver.h"
#include "config.h"
#include "formosa-hal/formosa-hal.h"
#include "formosa-llvm-util.h"
#include "formosa-memory.h"
#include "formosa-util.h"
#include "pocl-formosa-graph.h"
#include "pocl-formosa-internal.h"
#include "pocl_llvm.h"
#include "pocl_util.h"
#include "spirv_queries.h"

static inline uint64_t align(uint64_t n, size_t size) {
  return (n + size - 1) & ~(size - 1);
}

static struct pocl_work_graph_formosa_ops pocl_formosa_work_graph_formosa_ops =
    {.create_graph = pocl_formosa_create_work_graph,
     .create_node = pocl_formosa_create_work_graph_node,
     .create_edge = pocl_formosa_create_work_graph_edge,
     .get_info = pocl_formosa_get_work_graph_info,
     .free_graph = pocl_formosa_free_work_graph};

static cl_int formosa_copy_buf(void *data, pocl_mem_identifier *dst_mem_id,
                               cl_mem dst_buf, pocl_mem_identifier *src_mem_id,
                               cl_mem src_buf, size_t dst_offset,
                               size_t src_offset, size_t size);
cl_int pocl_formosa_alloc_subbuffer(cl_device_id device, cl_mem sub_buf);
void pocl_formosa_free_subbuffer(cl_device_id device, cl_mem sub_buf);
static void *formosa_copy_completion_thread(void *arg);
static cl_int formosa_submit_read_buf(void *data, void *__restrict__ host_ptr,
                                      pocl_mem_identifier *src_mem_id,
                                      cl_mem src_buf, size_t offset,
                                      size_t size, FsaCompletionToken *token);
static cl_int formosa_submit_write_buf(void *data,
                                       const void *__restrict__ host_ptr,
                                       pocl_mem_identifier *dst_mem_id,
                                       cl_mem dst_buf, size_t offset,
                                       size_t size, FsaCompletionToken *token);
static cl_int formosa_submit_copy_buf(
    void *data, pocl_mem_identifier *dst_mem_id, cl_mem dst_buf,
    pocl_mem_identifier *src_mem_id, cl_mem src_buf, size_t dst_offset,
    size_t src_offset, size_t size, FsaCompletionToken *token);

typedef struct {
  uint16_t dispatch_flags;
  size_t *local_sizes;
  size_t *num_groups;
  size_t *global_offsets;
  size_t local_mem_size;
  uintptr_t kernel_object;
  uintptr_t kernarg_address;
  uintptr_t kernel_trampoline;
  uintptr_t kernel_status;
} formosa_kernel_submit_args_t;

static FsaCommandSubmitStatus formosa_submit_kernel(
    void *context, FsaCompletionToken *token) {
  const formosa_kernel_submit_args_t *args =
      (const formosa_kernel_submit_args_t *)context;
  return fsa_cmd_start_kernel(args->dispatch_flags, args->local_sizes,
                              args->num_groups, args->global_offsets,
                              args->local_mem_size, args->kernel_object,
                              args->kernarg_address, args->kernel_trampoline,
                              args->kernel_status, token);
}

static cl_int formosa_validate_host_pointer(const void *host_ptr, size_t size) {
  return host_ptr == NULL && size != 0 ? CL_INVALID_VALUE : CL_SUCCESS;
}

void pocl_formosa_init_device_ops(struct pocl_device_ops *ops) {
  ops->device_name = "formosa";
  ops->build_hash = pocl_formosa_build_hash;
  ops->probe = pocl_formosa_probe;
  ops->init = pocl_formosa_init;
  ops->uninit = pocl_formosa_uninit;

  ops->init_context = pocl_formosa_init_context;
  ops->free_context = pocl_formosa_free_context;

  ops->run = pocl_formosa_run;
  ops->run_native = NULL;
  ops->run_work_graph_formosa = pocl_formosa_run_work_graph;

  ops->alloc_mem_obj = pocl_formosa_alloc_mem_obj;
  ops->free = pocl_formosa_free;
  ops->alloc_subbuffer = pocl_formosa_alloc_subbuffer;
  ops->free_subbuffer = pocl_formosa_free_subbuffer;

  ops->init_build = pocl_formosa_init_build;
  ops->build_source = pocl_driver_build_source;
  ops->link_program = pocl_driver_link_program;
  ops->build_binary = pocl_driver_build_binary;
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

  ops->set_kernel_stack_remap_formosa = pocl_formosa_set_kernel_stack_remap;
  ops->work_graph_formosa_ops = &pocl_formosa_work_graph_formosa_ops;
}

/*
 * Formosa stores an opaque host-side formosa_buffer_data_t in mem_ptr.  The
 * actual buffer payload lives in device memory at buf_address, so the generic
 * pocl_driver_copy() (which memcpy()s mem_ptr) cannot be used here.
 *
 * The unified HAL command is submitted through the firmware command stream.
 * The scheduler uses the submit_* variants below so completion is reported
 * by the event adapter rather than by this low-level callback.
 */
static cl_int formosa_copy_buf(void *data, pocl_mem_identifier *dst_mem_id,
                               cl_mem dst_buf, pocl_mem_identifier *src_mem_id,
                               cl_mem src_buf, size_t dst_offset,
                               size_t src_offset, size_t size) {
  (void)data;
  (void)dst_buf;
  (void)src_buf;

  formosa_memory_copy_addresses_t addresses = {};
  cl_int err = formosa_memory_resolve_copy_addresses(
      dst_mem_id, src_mem_id, dst_offset, src_offset, size, &addresses);
  if (err != CL_SUCCESS) return err;

  return formosa_memory_copy(kMemoryDomainDevice, addresses.src_addr,
                             kMemoryDomainDevice, addresses.dst_addr, size);
}

static cl_int formosa_submit_copy_buf(
    void *data, pocl_mem_identifier *dst_mem_id, cl_mem dst_buf,
    pocl_mem_identifier *src_mem_id, cl_mem src_buf, size_t dst_offset,
    size_t src_offset, size_t size, FsaCompletionToken *token) {
  (void)data;
  (void)dst_buf;
  (void)src_buf;
  formosa_memory_copy_addresses_t addresses = {};
  cl_int err = formosa_memory_resolve_copy_addresses(
      dst_mem_id, src_mem_id, dst_offset, src_offset, size, &addresses);
  if (err != CL_SUCCESS) return err;
  return formosa_memory_submit_copy(kMemoryDomainDevice, addresses.src_addr,
                                    kMemoryDomainDevice, addresses.dst_addr,
                                    size, token);
}

static cl_int formosa_submit_read_buf(void *data, void *__restrict__ host_ptr,
                                      pocl_mem_identifier *src_mem_id,
                                      cl_mem src_buf, size_t offset,
                                      size_t size, FsaCompletionToken *token) {
  (void)data;
  (void)src_buf;
  uint64_t device_addr = 0;
  cl_int err = formosa_memory_resolve_buffer_address(src_mem_id, offset, size,
                                                     &device_addr);
  if (err != CL_SUCCESS) return err;
  err = formosa_validate_host_pointer(host_ptr, size);
  if (err != CL_SUCCESS) return err;
  return formosa_memory_submit_copy(kMemoryDomainDevice, device_addr,
                                    kMemoryDomainHost, (uintptr_t)host_ptr,
                                    size, token);
}

static cl_int formosa_submit_write_buf(void *data,
                                       const void *__restrict__ host_ptr,
                                       pocl_mem_identifier *dst_mem_id,
                                       cl_mem dst_buf, size_t offset,
                                       size_t size, FsaCompletionToken *token) {
  (void)data;
  (void)dst_buf;
  uint64_t device_addr = 0;
  cl_int err = formosa_memory_resolve_buffer_address(dst_mem_id, offset, size,
                                                     &device_addr);
  if (err != CL_SUCCESS) return err;
  err = formosa_validate_host_pointer(host_ptr, size);
  if (err != CL_SUCCESS) return err;
  return formosa_memory_submit_copy(kMemoryDomainHost, (uintptr_t)host_ptr,
                                    kMemoryDomainDevice, device_addr, size,
                                    token);
}

/* Keep the generic device-op ABI usable for callers that invoke ops->copy
 * directly.  The Formosa scheduler uses formosa_copy_buf() so it can report
 * the error through the command event. */
void pocl_formosa_copy(void *data, pocl_mem_identifier *dst_mem_id,
                       cl_mem dst_buf, pocl_mem_identifier *src_mem_id,
                       cl_mem src_buf, size_t dst_offset, size_t src_offset,
                       size_t size) {
  cl_int err = formosa_copy_buf(data, dst_mem_id, dst_buf, src_mem_id, src_buf,
                                dst_offset, src_offset, size);
  if (err != CL_SUCCESS) POCL_MSG_ERR("pocl_formosa_copy: failed (%d)\n", err);
}

/**************************
 * Probe & Initialization *
 **************************/

static cl_bool formosa_available = CL_TRUE;
static char *formosa_build_hash = "formosa-riscv64-unknown-unknwon-elf";

void formosa_mark_unavailable(void) { formosa_available = CL_FALSE; }

static cl_int formosa_hal_error(cl_int fallback) {
  if (!fsa_hal_is_available()) {
    formosa_mark_unavailable();
    return CL_DEVICE_NOT_AVAILABLE;
  }
  return fallback;
}

unsigned int pocl_formosa_probe(struct pocl_device_ops *ops) {
  int err = fsa_probe();
  if (err != 0) {
    formosa_mark_unavailable();
    return 0;
  }
  return strncmp(ops->device_name, "formosa", 7) == 0;
}

char *pocl_formosa_build_hash(cl_device_id device) {
  char *res = (char *)calloc(strlen(formosa_build_hash) + 1, sizeof(char));
  strncpy(res, formosa_build_hash, strlen(formosa_build_hash));
  return res;
}

cl_int pocl_formosa_init(unsigned j, cl_device_id device,
                         const char *parameters) {
  pocl_formosa_data_t *dd;

  assert(device->data == NULL);

  pocl_init_default_device_infos(device, FORMOSA_DEVICE_EXTENSIONS);
  device->features = FORMOSA_DEVICE_FEATURES_30;

  if (strstr(FORMOSA_DEVICE_EXTENSIONS, "cl_khr_kernel_clock") != NULL) {
    device->kernel_clock_caps = CL_DEVICE_KERNEL_CLOCK_SCOPE_DEVICE_KHR |
                                CL_DEVICE_KERNEL_CLOCK_SCOPE_WORK_GROUP_KHR |
                                CL_DEVICE_KERNEL_CLOCK_SCOPE_SUB_GROUP_KHR;
  }

  SETUP_DEVICE_CL_VERSION(device, FORMOSA_DEVICE_CL_VERSION_MAJOR,
                          FORMOSA_DEVICE_CL_VERSION_MINOR);

  dd = (pocl_formosa_data_t *)calloc(1, sizeof(pocl_formosa_data_t));
  if (dd == NULL) {
    return CL_OUT_OF_HOST_MEMORY;
  }

  device->vendor = "CASLab";
  device->long_name = "Formosa Open RISC-V ML-Oriented SIMT Architecture";
  device->short_name = "FORMOSA";
  device->vendor_id = 0;
  device->type = CL_DEVICE_TYPE_GPU;

  device->spmd = CL_TRUE;
  device->run_workgroup_pass = CL_FALSE;
  device->execution_capabilities = CL_EXEC_KERNEL;
  device->autolocals_to_args = POCL_AUTOLOCALS_TO_ARGS_ALWAYS;
  device->device_alloca_locals = CL_FALSE;
  device->device_side_printf = CL_TRUE;
  device->printf_buffer_size = PRINTF_BUFFER_SIZE * 1024;
  device->has_64bit_long = CL_TRUE;

  device->address_bits = 64;
  device->llvm_target_triplet = "riscv64-unknown-unknown-elf";
  device->llvm_abi = "lp64";
  device->llvm_cpu = "formosa-gpgpu";
  device->kernellib_name = "kernel-riscv64-formosa";
  device->kernellib_fallback_name = NULL;
  device->kernellib_subdir = "formosa";

  device->image_support = CL_FALSE;

#if defined(ENABLE_SPIRV)
  device->supported_spir_v_versions =
      "SPIR-V_1.5 SPIR-V_1.4 SPIR-V_1.3 SPIR-V_1.2 SPIR-V_1.1 SPIR-V_1.0";
  device->supported_spirv_extensions = "+SPV_KHR_no_integer_wrap_decoration";
#endif

  size_t num_warps = fsa_warps_per_core();
  size_t num_threads = fsa_threads_per_warp();
  uint64_t max_work_group_size = num_warps * num_threads;

  device->global_mem_cache_type = CL_READ_WRITE_CACHE;
  device->global_mem_cacheline_size = fsa_cache_block_size();
  device->global_mem_cache_size = fsa_cache_size();
  device->global_mem_size = fsa_global_mem_size();
  device->max_mem_alloc_size = fsa_global_mem_size();
  device->local_mem_size = fsa_local_mem_size();
  device->max_work_group_size = max_work_group_size;
  device->max_work_item_sizes[0] = max_work_group_size;
  device->max_work_item_sizes[1] = max_work_group_size;
  device->max_work_item_sizes[2] = max_work_group_size;
  device->max_compute_units = 1;
  device->mem_base_addr_align = 128;  // TODO: determine this

  pocl_setup_extensions_with_version(device);
  pocl_setup_ils_with_version(device);
  pocl_setup_builtin_kernels_with_version(device);
  pocl_setup_spirv_queries(device);

  dd->context_ref_count = 0;

  POCL_INIT_LOCK(dd->compile_lock);
  POCL_INIT_LOCK(dd->cq_lock);
  POCL_INIT_LOCK(dd->copy_lock);
  POCL_INIT_COND(dd->copy_cond);

  device->data = dd;
  device->available = &formosa_available;

  dd->kernel_buffer = NULL;
  formosa_available = CL_TRUE;

  int err = fsa_hal_init();
  if (err != 0) {
    formosa_available = CL_FALSE;
    POCL_DESTROY_LOCK(dd->compile_lock);
    POCL_DESTROY_LOCK(dd->cq_lock);
    POCL_DESTROY_COND(dd->copy_cond);
    POCL_DESTROY_LOCK(dd->copy_lock);
    POCL_MEM_FREE(dd);
    device->data = NULL;
    POCL_MSG_ERR("pocl_formosa_init: HAL initialization failed (%d)\n", err);
    return CL_DEVICE_NOT_AVAILABLE;
  }

  dd->copy_thread_stop = CL_FALSE;
  dd->copy_thread_started = CL_TRUE;
  POCL_CREATE_THREAD(dd->copy_thread, formosa_copy_completion_thread, dd);

  return CL_SUCCESS;
}

cl_int pocl_formosa_uninit(unsigned j, cl_device_id device) {
  pocl_formosa_data_t *dd = (pocl_formosa_data_t *)device->data;
  if (dd == NULL) return CL_SUCCESS;

  if (dd->copy_thread_started) {
    POCL_LOCK(dd->copy_lock);
    dd->copy_thread_stop = CL_TRUE;
    POCL_BROADCAST_COND(dd->copy_cond);
    POCL_UNLOCK(dd->copy_lock);
    POCL_JOIN_THREAD(dd->copy_thread);
    dd->copy_thread_started = CL_FALSE;
  }
  while (dd->copy_pending != NULL) {
    formosa_pending_copy_t *pending = dd->copy_pending;
    dd->copy_pending = pending->next;
    (void)fsa_release_completion(pending->token);
    POCL_MEM_FREE(pending);
  }
  dd->copy_pending_tail = NULL;

  POCL_DESTROY_LOCK(dd->compile_lock);
  POCL_DESTROY_LOCK(dd->cq_lock);
  POCL_DESTROY_COND(dd->copy_cond);
  POCL_DESTROY_LOCK(dd->copy_lock);
  if (fsa_hal_cleanup() != 0) {
    POCL_MSG_ERR("pocl_formosa_uninit: HAL cleanup failed\n");
  }
  if (dd->kernel_buffer != NULL) {
    POCL_MEM_FREE(dd->kernel_buffer);
  }
  POCL_MEM_FREE(device->data);
  device->data = NULL;
  return CL_SUCCESS;
}

/* Execute an NDRange on Formosa. Returns an OpenCL error code; does not
 * update the command event (the Formosa scheduler does that). */
static cl_int formosa_run_kernel(void *data, _cl_command_node *cmd) {
  pocl_formosa_data_t *dd;
  cl_uint device_i = cmd->program_device_i;
  cl_kernel kernel = cmd->command.run.kernel;
  cl_program program = kernel->program;
  pocl_kernel_metadata_t *meta = kernel->meta;
  struct pocl_context *pc = &cmd->command.run.pc;
  cl_int errcode = CL_SUCCESS;
  int err = 0;

  uint8_t *host_kargs_base_ptr = NULL;
  void *device_args_buffer_addr = NULL;
  void *device_kernel_status_addr = NULL;
  void *device_printf_buffer_addr = NULL;
  void *device_printf_position_addr = NULL;
  void *device_kernel_addr = NULL;
  char *trampoline_name = NULL;
  char *host_printf_buffer = NULL;

  uint32_t num_groups = 1;
  uint32_t group_size = 1;
  for (uint32_t i = 0; i < pc->work_dim; ++i) {
    num_groups *= pc->num_groups[i];
    group_size *= pc->local_size[i];
  }
  if (num_groups == 0 || group_size == 0) return CL_SUCCESS;

  assert(data != NULL);
  dd = (pocl_formosa_data_t *)data;

  const uint32_t ptr_size = 8;
  const uint32_t word_size = 8;
  const uint32_t printf_meta_size = sizeof(formosa_printf_launch_meta_t);

  // calculate kernel arguments buffer size
  uint64_t local_mem_size = 0;  // total local memory size

  // kernel arguments buffer size
  // it contains:
  // 0. PoCL device-side printf metadata consumed by firmware/CP
  // 1. local memory size (8 bytes)
  // 2. other arguments
  size_t kargs_buffer_size = printf_meta_size + word_size;

  for (int i = 0; i < meta->num_args; ++i) {
    struct pocl_argument *al = &(cmd->command.run.arguments[i]);
    if (ARG_IS_LOCAL(meta->arg_info[i])) {
      local_mem_size += al->size;
      // add space for local memory offset
      kargs_buffer_size = align(kargs_buffer_size, word_size) + word_size;
    } else if ((meta->arg_info[i].type == POCL_ARG_TYPE_POINTER) ||
               (meta->arg_info[i].type == POCL_ARG_TYPE_IMAGE) ||
               (meta->arg_info[i].type == POCL_ARG_TYPE_SAMPLER)) {
      kargs_buffer_size = align(kargs_buffer_size, ptr_size) + ptr_size;
    } else {
      // scalar argument
      kargs_buffer_size = align(kargs_buffer_size, al->size) + al->size;
    }
  }

  // local buffers
  for (int i = 0; i < meta->num_locals; ++i) {
    local_mem_size += meta->local_sizes[i];
    kargs_buffer_size = align(kargs_buffer_size, word_size) + word_size;
  }

  // check occupancy
  uint64_t available_local_mem = 0;
  err = pocl_fsa_check_occupancy(group_size, local_mem_size,
                                 &available_local_mem);
  if (err != 0) {
    POCL_MSG_ERR(
        "pocl_formosa_run: occupancy check failed "
        "(group_size=%" PRIu32 ", local_mem_size=%" PRIu64
        ", available_local_mem=%" PRIu64 ")\n",
        group_size, local_mem_size, available_local_mem);
    errcode = CL_INVALID_WORK_GROUP_SIZE;
    goto FAIL;
  }

  // allocate arguments host buffer
  host_kargs_base_ptr = (uint8_t *)malloc(kargs_buffer_size);
  if (host_kargs_base_ptr == NULL) {
    errcode = CL_OUT_OF_HOST_MEMORY;
    goto FAIL;
  }
  memset(host_kargs_base_ptr, 0, kargs_buffer_size);

  // allocate kernel arguments buffer
  formosa_buffer_data_t fsa_kargs_buffer;
  memset(&fsa_kargs_buffer, 0, sizeof(formosa_buffer_data_t));
  err = fsa_malloc(&device_args_buffer_addr, kargs_buffer_size);
  if (err != 0) {
    POCL_MSG_ERR("pocl_formosa_run: device kargs allocation failed\n");
    errcode = CL_OUT_OF_RESOURCES;
    goto FAIL;
  }
  err = fsa_malloc(&device_kernel_status_addr, sizeof(KernelStatus));
  if (err != 0) {
    POCL_MSG_ERR("pocl_formosa_run: device kernel status allocation failed\n");
    errcode = CL_OUT_OF_RESOURCES;
    goto FAIL;
  }
  err = fsa_malloc(&device_printf_buffer_addr, cmd->device->printf_buffer_size);
  if (err != 0) {
    POCL_MSG_ERR("pocl_formosa_run: device printf buffer allocation failed\n");
    errcode = CL_OUT_OF_RESOURCES;
    goto FAIL;
  }
  POCL_MSG_PRINT_INFO(
      "Device printf buffer allocated at address %p with size %zu "
      "bytes\n",
      device_printf_buffer_addr, cmd->device->printf_buffer_size);
  err = fsa_malloc(&device_printf_position_addr, sizeof(uint32_t));
  if (err != 0) {
    POCL_MSG_ERR(
        "pocl_formosa_run: device printf position allocation failed\n");
    errcode = CL_OUT_OF_RESOURCES;
    goto FAIL;
  }
  fsa_kargs_buffer.buf_address = (uint64_t)device_args_buffer_addr;
  fsa_kargs_buffer.buf_size = kargs_buffer_size;

  // write arguments
  formosa_printf_launch_meta_t printf_meta = {
      .printf_buffer = (uint64_t)device_printf_buffer_addr,
      .printf_buffer_position = (uint64_t)device_printf_position_addr,
      .printf_buffer_capacity = (uint32_t)cmd->device->printf_buffer_size,
      .reserved = 0,
  };
  memcpy(host_kargs_base_ptr, &printf_meta, sizeof(printf_meta));

  uint32_t zero_printf_position = 0;
  err = fsa_copy_to_dev((uintptr_t)device_printf_position_addr,
                        &zero_printf_position, sizeof(zero_printf_position));
  if (err != 0) {
    POCL_MSG_ERR("pocl_formosa_run: device printf position reset failed\n");
    errcode = formosa_hal_error(CL_OUT_OF_RESOURCES);
    goto FAIL;
  }

  uint32_t host_args_offset = printf_meta_size;
  uint64_t local_mem_offset = 0;

  if (local_mem_size > 0) {
    memcpy(host_kargs_base_ptr + host_args_offset, &local_mem_size,
           word_size);  // total local memory size
    host_args_offset += word_size;
  } else {
    // if no local memory, write 0
    memset(host_kargs_base_ptr + host_args_offset, 0,
           word_size);  // local size
    host_args_offset += word_size;
  }

  for (int i = 0; i < meta->num_args; ++i) {
    struct pocl_argument *al = &(cmd->command.run.arguments[i]);
    if (ARG_IS_LOCAL(meta->arg_info[i])) {
      host_args_offset = align(host_args_offset, word_size);
      memcpy(host_kargs_base_ptr + host_args_offset, &local_mem_offset,
             word_size);  // local memory offset
      host_args_offset += word_size;
      local_mem_offset += al->size;
    } else if (meta->arg_info[i].type == POCL_ARG_TYPE_POINTER) {
      if (al->value == NULL) {
        host_args_offset = align(host_args_offset, ptr_size);
        memset(host_kargs_base_ptr + host_args_offset, 0,
               ptr_size);  // NULL pointer value
        host_args_offset += ptr_size;
      } else {
        cl_mem m = (*(cl_mem *)(al->value));
        formosa_buffer_data_t *buf_data =
            (formosa_buffer_data_t *)m->device_ptrs[cmd->device->global_mem_id]
                .mem_ptr;
        if (buf_data == NULL) {
          POCL_MSG_ERR(
              "pocl_formosa_run: buffer argument has no device allocation\n");
          errcode = CL_INVALID_MEM_OBJECT;
          goto FAIL;
        }
        uint64_t dev_mem_addr = buf_data->buf_address + al->offset;
        host_args_offset = align(host_args_offset, word_size);
        memcpy(host_kargs_base_ptr + host_args_offset, &dev_mem_addr,
               ptr_size);  // pointer value
        host_args_offset += ptr_size;
      }
    } else if (meta->arg_info[i].type == POCL_ARG_TYPE_IMAGE) {
      POCL_MSG_ERR("pocl_formosa_run: image arguments are not supported\n");
      errcode = CL_INVALID_KERNEL_ARGS;
      goto FAIL;
    } else if (meta->arg_info[i].type == POCL_ARG_TYPE_SAMPLER) {
      POCL_MSG_ERR("pocl_formosa_run: sampler arguments are not supported\n");
      errcode = CL_INVALID_KERNEL_ARGS;
      goto FAIL;
    } else {
      // scalar argument
      if (al->value == NULL) {
        POCL_MSG_ERR("pocl_formosa_run: missing scalar kernel argument\n");
        errcode = CL_INVALID_KERNEL_ARGS;
        goto FAIL;
      }
      host_args_offset = align(host_args_offset, al->size);
      memcpy(host_kargs_base_ptr + host_args_offset, al->value,
             al->size);  // scalar value
      host_args_offset += al->size;
    }
  }

  for (int i = 0; i < meta->num_locals; ++i) {
    host_args_offset = align(host_args_offset, word_size);
    memcpy(host_kargs_base_ptr + host_args_offset, &local_mem_offset,
           word_size);  // arg offset
    host_args_offset += word_size;
    local_mem_offset += meta->local_sizes[i];
  }

  // upload kernel arguments buffer
  err = fsa_copy_to_dev(fsa_kargs_buffer.buf_address, host_kargs_base_ptr,
                        kargs_buffer_size);
  if (err != 0) {
    POCL_MSG_ERR("pocl_formosa_run: kernel argument copy to device failed\n");
    errcode = formosa_hal_error(CL_OUT_OF_RESOURCES);
    goto FAIL;
  }

  // release argument host buffer
  free(host_kargs_base_ptr);
  host_kargs_base_ptr = NULL;

  // upload kernel to device
  uintptr_t entry_pc = 0, trampoline_pc = 0;
  if (dd->kernel_buffer == NULL) {
    char sz_program_fsabin[POCL_MAX_PATHNAME_LENGTH];
    err = pocl_fsa_get_elf_name(program, device_i, sz_program_fsabin);
    if (err != 0) {
      POCL_MSG_ERR("pocl_formosa_run: failed to resolve kernel ELF path\n");
      errcode = CL_INVALID_PROGRAM_EXECUTABLE;
      goto FAIL;
    }
    POCL_MSG_PRINT_INFO("elf path: %s\n", sz_program_fsabin);
    {
      uint64_t kernel_dev_addr = 0;
      err = pocl_fsa_upload_kernel(sz_program_fsabin, &kernel_dev_addr);
      if (err != 0) {
        POCL_MSG_ERR("pocl_formosa_run: kernel upload failed\n");
        errcode = formosa_hal_error(CL_OUT_OF_RESOURCES);
        goto FAIL;
      }
      device_kernel_addr = (void *)(uintptr_t)kernel_dev_addr;
    }

    /* _start is placed at .org 0x0 — address 0 is valid. UINT64_MAX means
     * lookup failure. */
    uint64_t start_off = pocl_fsa_get_symbol_pc(sz_program_fsabin, "_start");
    if (start_off == UINT64_MAX) {
      POCL_MSG_ERR("pocl_formosa_run: _start symbol not found in kernel ELF\n");
      errcode = CL_INVALID_PROGRAM_EXECUTABLE;
      goto FAIL;
    }
    entry_pc = start_off + (uintptr_t)device_kernel_addr;
    POCL_MSG_PRINT_INFO("entry_pc: %lx\n", entry_pc);

    trampoline_name = malloc(strlen(kernel->name) + 13);
    if (trampoline_name == NULL) {
      errcode = CL_OUT_OF_HOST_MEMORY;
      goto FAIL;
    }
    sprintf(trampoline_name, "%s_trampolined", kernel->name);
    uint64_t tramp_off =
        pocl_fsa_get_symbol_pc(sz_program_fsabin, trampoline_name);
    if (tramp_off == UINT64_MAX) {
      POCL_MSG_ERR("pocl_formosa_run: trampoline symbol %s not found\n",
                   trampoline_name);
      errcode = CL_INVALID_PROGRAM_EXECUTABLE;
      goto FAIL;
    }
    trampoline_pc = tramp_off + (uintptr_t)device_kernel_addr;
    POCL_MSG_PRINT_INFO("trampoline_pc: %lx\n", trampoline_pc);
    free(trampoline_name);
    trampoline_name = NULL;
  }

  // launch kernel execution
  FsaCompletionToken token = 0;
  uint16_t dispatch_flags = pc->work_dim | FSA_KERNEL_DISPATCH_HAS_PRINTF_META;
  if (pocl_formosa_kernel_stack_remap_enabled(kernel, cmd->device))
    dispatch_flags |= FSA_KERNEL_DISPATCH_STACK_REMAP;
  const formosa_kernel_submit_args_t submit_args = {
      dispatch_flags,
      pc->local_size,
      pc->num_groups,
      pc->global_offset,
      local_mem_size,
      entry_pc,
      (uintptr_t)device_args_buffer_addr,
      (uintptr_t)trampoline_pc,
      (uintptr_t)device_kernel_status_addr};
  const FsaCommandSubmitStatus submit_status =
      pocl_fsa_submit_with_backpressure(formosa_submit_kernel,
                                        (void *)&submit_args, &token);

  if (submit_status != kFsaCommandSubmitAccepted) {
    POCL_MSG_ERR("pocl_formosa_run: kernel launch failed\n");
    errcode = formosa_hal_error(CL_OUT_OF_RESOURCES);
    goto FAIL;
  }

  /* Wait for execution to complete; terminal wait releases the token. */
  errcode =
      pocl_fsa_wait_completion(token, (uintptr_t)device_kernel_status_addr);
  if (errcode != CL_SUCCESS) {
    POCL_MSG_ERR("pocl_formosa_run: kernel execution failed\n");
    if (errcode == CL_DEVICE_NOT_AVAILABLE) formosa_mark_unavailable();
    goto FAIL;
  }

  uint32_t printf_position = 0;
  err = fsa_copy_from_dev((uintptr_t)device_printf_position_addr,
                          &printf_position, sizeof(printf_position));
  if (err != 0) {
    POCL_MSG_ERR(
        "pocl_formosa_run: reading printf position from device failed\n");
    errcode = formosa_hal_error(CL_OUT_OF_RESOURCES);
    goto FAIL;
  }
  if (printf_position > cmd->device->printf_buffer_size) {
    POCL_MSG_ERR(
        "pocl_formosa_run: invalid printf buffer position %u (capacity %zu)\n",
        printf_position, cmd->device->printf_buffer_size);
    errcode = CL_OUT_OF_RESOURCES;
    goto FAIL;
  }
  if (printf_position > 0) {
    host_printf_buffer = (char *)malloc(printf_position);
    if (host_printf_buffer == NULL) {
      errcode = CL_OUT_OF_HOST_MEMORY;
      goto FAIL;
    }
    err = fsa_copy_from_dev((uintptr_t)device_printf_buffer_addr,
                            host_printf_buffer, printf_position);
    if (err != 0) {
      POCL_MSG_ERR(
          "pocl_formosa_run: reading printf buffer from device failed\n");
      errcode = formosa_hal_error(CL_OUT_OF_RESOURCES);
      goto FAIL;
    }
    pocl_write_printf_buffer(host_printf_buffer, printf_position);
    free(host_printf_buffer);
    host_printf_buffer = NULL;
  }

  // release arguments device buffer
  if (device_args_buffer_addr && fsa_free(device_args_buffer_addr) != 0)
    POCL_MSG_ERR("pocl_formosa_run: kernel argument free failed\n");
  // release kernel status device buffer
  if (device_kernel_status_addr && fsa_free(device_kernel_status_addr) != 0)
    POCL_MSG_ERR("pocl_formosa_run: kernel status free failed\n");
  if (device_printf_buffer_addr && fsa_free(device_printf_buffer_addr) != 0)
    POCL_MSG_ERR("pocl_formosa_run: printf buffer free failed\n");
  if (device_printf_position_addr && fsa_free(device_printf_position_addr) != 0)
    POCL_MSG_ERR("pocl_formosa_run: printf position free failed\n");
  return CL_SUCCESS;

FAIL:
  free(host_kargs_base_ptr);
  free(trampoline_name);
  free(host_printf_buffer);
  if (device_args_buffer_addr) fsa_free(device_args_buffer_addr);
  if (device_kernel_status_addr) fsa_free(device_kernel_status_addr);
  if (device_printf_buffer_addr) fsa_free(device_printf_buffer_addr);
  if (device_printf_position_addr) fsa_free(device_printf_position_addr);
  if (device_kernel_addr) fsa_free(device_kernel_addr);
  return errcode;
}

void pocl_formosa_run(void *data, _cl_command_node *cmd) {
  /* Prefer formosa_command_scheduler for NDRange (it fails/completes the
   * event). This ops entry is only used if something calls ops->run directly.
   */
  cl_int err = formosa_run_kernel(data, cmd);
  if (err != CL_SUCCESS)
    POCL_MSG_ERR("pocl_formosa_run: kernel execution failed (%d)\n", err);
}

/**************************
 * Program                *
 **************************/

char *pocl_formosa_init_build(void *data) {
  // clang -cc1 options
  const char *pocl_cflags = pocl_get_string_option("POCL_FORMOSA_CFLAGS", "");
  if (strstr(pocl_cflags, "-fsa-ics-first") != NULL)
    return strdup(
        "-target-feature +zaamo "
        "-fsa-ics-first-cg -target-feature +xformosapri");
  return strdup("-target-feature +zaamo");
}

int pocl_formosa_post_build_program(cl_program program, cl_uint device_i) {
  cl_device_id dev = program->devices[device_i];
  pocl_formosa_data_t *ddata = (pocl_formosa_data_t *)dev->data;
  formosa_program_data_t *pdata = NULL;

  POCL_LOCK(ddata->compile_lock);

  int err = pocl_llvm_run_passes_on_program(program, device_i);
  if (err != CL_SUCCESS) {
    POCL_MSG_ERR("LLVM passes failed for program\n");
    goto POST_BUILD_PROGRAM_FINALLY;
  }

  pdata = (formosa_program_data_t *)calloc(1, sizeof(formosa_program_data_t));
  pdata->kernel_names = NULL;

  char fsa_program_bin[POCL_MAX_PATHNAME_LENGTH];
  err = pocl_fsa_get_elf_name(program, device_i, fsa_program_bin);
  if (err != 0) {
    POCL_MSG_ERR("Get ELF name failed\n");
    goto POST_BUILD_PROGRAM_FINALLY;
  }
  err = pocl_fsa_compile_program(&pdata->kernel_names, &pdata->num_kernels,
                                 fsa_program_bin, program->compiler_options,
                                 program->llvm_irs[device_i]);

POST_BUILD_PROGRAM_FINALLY:
  program->data[device_i] = pdata;

  POCL_UNLOCK(ddata->compile_lock);

  return err;
}

int pocl_formosa_free_program(cl_device_id device, cl_program program,
                              unsigned program_device_i) {
  pocl_formosa_data_t *dd = (pocl_formosa_data_t *)device->data;
  formosa_program_data_t *pdata =
      (formosa_program_data_t *)program->data[program_device_i];
  if (pdata == NULL) return CL_SUCCESS;

  pocl_driver_free_program(device, program, program_device_i);

  POCL_MEM_FREE(pdata->kernel_names);
  POCL_MEM_FREE(pdata);
  program->data[program_device_i] = NULL;

  return CL_SUCCESS;
}

/**************************
 * Kernel                 *
 **************************/

cl_int pocl_formosa_create_kernel(cl_device_id device, cl_program program,
                                  cl_kernel kernel, unsigned program_device_i) {
  pocl_kernel_metadata_t *meta = kernel->meta;
  assert(meta->data != NULL);
  assert(kernel->data != NULL);
  assert(kernel->data[program_device_i] == NULL);

  formosa_kernel_instance_data_t *instance_data =
      (formosa_kernel_instance_data_t *)calloc(1, sizeof(*instance_data));
  if (instance_data == NULL) return CL_OUT_OF_HOST_MEMORY;
  /* LSAR is enabled by default; CL_FALSE can opt this kernel out. */
  instance_data->stack_remap = CL_TRUE;
  kernel->data[program_device_i] = instance_data;

  // device-specific kernel metadata
  formosa_kernel_data_t *kdata =
      (formosa_kernel_data_t *)meta->data[program_device_i];
  if (kdata != NULL) {
    ++kdata->ref_count;
    return CL_SUCCESS;
  }

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
  if (kdata == NULL) {
    POCL_MEM_FREE(kernel->data[program_device_i]);
    return CL_OUT_OF_HOST_MEMORY;
  }
  kdata->kernel_id = i;
  ++kdata->ref_count;

  meta->data[program_device_i] = kdata;

  return CL_SUCCESS;
}

cl_int pocl_formosa_free_kernel(cl_device_id device, cl_program program,
                                cl_kernel kernel, unsigned program_device_i) {
  pocl_kernel_metadata_t *meta = kernel->meta;
  assert(meta->data != NULL);

  if (kernel->data != NULL) POCL_MEM_FREE(kernel->data[program_device_i]);

  formosa_kernel_data_t *kdata =
      (formosa_kernel_data_t *)meta->data[program_device_i];
  if (kdata == NULL) return CL_SUCCESS;

  --kdata->ref_count;
  if (kdata->ref_count == 0) {
    POCL_MEM_FREE(kdata);
    meta->data[program_device_i] = NULL;
  }

  return CL_SUCCESS;
}

cl_int pocl_formosa_set_kernel_stack_remap(cl_device_id device,
                                           unsigned program_device_i,
                                           cl_kernel kernel,
                                           cl_bool designate) {
  if (device == NULL || kernel == NULL || kernel->program == NULL ||
      kernel->data == NULL ||
      program_device_i >= kernel->program->num_devices ||
      pocl_real_dev(kernel->program->devices[program_device_i]) != device ||
      kernel->data[program_device_i] == NULL)
    return CL_INVALID_KERNEL;

  formosa_kernel_instance_data_t *instance_data =
      (formosa_kernel_instance_data_t *)kernel->data[program_device_i];
  instance_data->stack_remap = designate;
  return CL_SUCCESS;
}

cl_bool pocl_formosa_kernel_stack_remap_enabled(cl_kernel kernel,
                                                cl_device_id device) {
  if (kernel == NULL || kernel->program == NULL || kernel->data == NULL ||
      device == NULL)
    return CL_FALSE;

  cl_device_id realdev = pocl_real_dev(device);
  for (cl_uint i = 0; i < kernel->program->num_devices; ++i) {
    if (pocl_real_dev(kernel->program->devices[i]) != realdev) continue;

    formosa_kernel_instance_data_t *instance_data =
        (formosa_kernel_instance_data_t *)kernel->data[i];
    return instance_data != NULL && instance_data->stack_remap;
  }
  return CL_FALSE;
}

/**************************
 * Context                *
 **************************/

cl_int pocl_formosa_init_context(cl_device_id device, cl_context context) {
  pocl_formosa_data_t *dd = (pocl_formosa_data_t *)device->data;
  if (dd == NULL) return CL_SUCCESS;

  dd->context_ref_count++;

  return CL_SUCCESS;
}

cl_int pocl_formosa_free_context(cl_device_id device, cl_context context) {
  pocl_formosa_data_t *dd = (pocl_formosa_data_t *)device->data;
  if (dd == NULL) return CL_SUCCESS;

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
                                  void *host_ptr) {
  pocl_mem_identifier *p = &mem_obj->device_ptrs[device->global_mem_id];

  cl_mem_flags flags = mem_obj->flags;
  assert((flags & (CL_MEM_READ_WRITE | CL_MEM_WRITE_ONLY | CL_MEM_READ_ONLY)) !=
         0);

  void *addr;
  int err = fsa_malloc(&addr, mem_obj->size);
  if (err) {
    return CL_MEM_OBJECT_ALLOCATION_FAILURE;
  }

  formosa_buffer_data_t *temp = malloc(sizeof(formosa_buffer_data_t));
  if (temp == NULL) {
    if (fsa_free((void *)addr) != 0) {
      POCL_MSG_ERR(
          "pocl_formosa_alloc_mem_obj: failed to free device memory after "
          "host allocation failure\n");
    }
    return CL_OUT_OF_HOST_MEMORY;
  }
  memset(temp, 0, sizeof(formosa_buffer_data_t));
  if (host_ptr) {  // READ_WRITE, WRITE_ONLY
    temp->buf_address = (uint64_t)addr;
    temp->buf_size = mem_obj->size;
    temp->msg_id = 0;
    err =
        formosa_memory_copy(kMemoryDomainHost, (uintptr_t)host_ptr,
                            kMemoryDomainDevice, (uint64_t)addr, mem_obj->size);
    if (err != 0) {
      fsa_free((void *)temp->buf_address);
      POCL_MEM_FREE(temp);
      return err;
    }
  } else {  // READ_ONLY
    temp->buf_address = (uint64_t)addr;
    temp->buf_size = mem_obj->size;
    temp->msg_id = 0;
  }

  p->mem_ptr = temp;
  return CL_SUCCESS;
}

void pocl_formosa_free(cl_device_id device, cl_mem mem_obj) {
  pocl_mem_identifier *p = &mem_obj->device_ptrs[device->global_mem_id];
  cl_mem_flags flags = mem_obj->flags;
  formosa_buffer_data_t *fb = (formosa_buffer_data_t *)p->mem_ptr;
  if (!fb) {
    POCL_MSG_ERR("pocl_formosa_free: buffer has no device allocation\n");
    return;
  }
  if (flags & CL_MEM_ALLOC_HOST_PTR) {
    pocl_release_mem_host_ptr(mem_obj);
  }
  if (fsa_free((void *)fb->buf_address) != 0) {
    POCL_MSG_ERR("pocl_formosa_free: device free failed for buffer %p\n",
                 (void *)fb->buf_address);
  }
  POCL_MEM_FREE(fb);
  p->mem_ptr = NULL;
}

/* A sub-buffer aliases its parent allocation.  PoCL may pre-populate
 * mem_ptr with parent metadata plus origin, which is not a
 * formosa_buffer_data_t.  Materialize a small metadata object so every copy
 * callback receives the same deep representation and no callback has to know
 * about PoCL's internal pointer encoding. */
cl_int pocl_formosa_alloc_subbuffer(cl_device_id device, cl_mem sub_buf) {
  if (sub_buf == NULL || sub_buf->parent == NULL) return CL_INVALID_MEM_OBJECT;
  pocl_mem_identifier *p = &sub_buf->device_ptrs[device->global_mem_id];
  pocl_mem_identifier *parent_p =
      &sub_buf->parent->device_ptrs[device->global_mem_id];
  formosa_buffer_data_t *parent_data =
      (formosa_buffer_data_t *)parent_p->mem_ptr;
  if (parent_data == NULL || sub_buf->origin > parent_data->buf_size ||
      sub_buf->size > parent_data->buf_size - sub_buf->origin)
    return CL_INVALID_VALUE;

  formosa_buffer_data_t *sub_data =
      (formosa_buffer_data_t *)calloc(1, sizeof(*sub_data));
  if (sub_data == NULL) return CL_OUT_OF_HOST_MEMORY;
  sub_data->buf_address = parent_data->buf_address + sub_buf->origin;
  sub_data->buf_size = sub_buf->size;
  sub_data->msg_id = parent_data->msg_id;
  p->mem_ptr = sub_data;
  return CL_SUCCESS;
}

void pocl_formosa_free_subbuffer(cl_device_id device, cl_mem sub_buf) {
  if (sub_buf == NULL) return;
  pocl_mem_identifier *p = &sub_buf->device_ptrs[device->global_mem_id];
  POCL_MEM_FREE(p->mem_ptr);
  p->mem_ptr = NULL;
}

/**************************
 * Event Handling         *
 **************************/

static cl_int formosa_read_buf(void *data, void *__restrict__ host_ptr,
                               pocl_mem_identifier *src_mem_id, cl_mem src_buf,
                               size_t offset, size_t size) {
  uint64_t device_addr = 0;
  cl_int err = formosa_memory_resolve_buffer_address(src_mem_id, offset, size,
                                                     &device_addr);
  if (err != CL_SUCCESS) return err;
  err = formosa_validate_host_pointer(host_ptr, size);
  if (err != CL_SUCCESS) return err;
  cl_int copy_status =
      formosa_memory_copy(kMemoryDomainDevice, device_addr, kMemoryDomainHost,
                          (uintptr_t)host_ptr, size);
  if (copy_status != CL_SUCCESS) {
    POCL_MSG_ERR("pocl_formosa_read: copy from device failed\n");
    return copy_status;
  }
  return CL_SUCCESS;
}

static cl_int formosa_write_buf(void *data, const void *__restrict__ host_ptr,
                                pocl_mem_identifier *dst_mem_id, cl_mem dst_buf,
                                size_t offset, size_t size) {
  uint64_t device_addr = 0;
  cl_int err = formosa_memory_resolve_buffer_address(dst_mem_id, offset, size,
                                                     &device_addr);
  if (err != CL_SUCCESS) return err;
  err = formosa_validate_host_pointer(host_ptr, size);
  if (err != CL_SUCCESS) return err;
  cl_int copy_status =
      formosa_memory_copy(kMemoryDomainHost, (uintptr_t)host_ptr,
                          kMemoryDomainDevice, device_addr, size);
  if (copy_status != CL_SUCCESS) {
    POCL_MSG_ERR("pocl_formosa_write: copy to device failed\n");
    return copy_status;
  }
  return CL_SUCCESS;
}

/* Formosa-local finish: on error mark the event failed (negative OpenCL
 * status) instead of completing as success. Do not use pocl_exec_command for
 * these types — it always COMPLETE after ops->run/read/write. */
static void formosa_finish_command(cl_event event, cl_int err,
                                   const char *ok_msg, const char *fail_msg) {
  if (err != CL_SUCCESS)
    POCL_UPDATE_EVENT_FAILED_MSG(err, event, fail_msg);
  else
    POCL_UPDATE_EVENT_COMPLETE_MSG(event, ok_msg);
}

static void formosa_enqueue_pending_copy(pocl_formosa_data_t *dd,
                                         formosa_pending_copy_t *pending) {
  POCL_LOCK(dd->copy_lock);
  /* The Formosa command stream is globally in-order, so append records in
   * submission order.  Firmware serializes the actual copy phases, while
   * PoCL may keep submitting later commands and let the completion worker
   * retire records independently. */
  pending->next = NULL;
  if (dd->copy_pending_tail != NULL)
    dd->copy_pending_tail->next = pending;
  else
    dd->copy_pending = pending;
  dd->copy_pending_tail = pending;
  POCL_SIGNAL_COND(dd->copy_cond);
  POCL_UNLOCK(dd->copy_lock);
}

static void formosa_submit_memory_copy_command(pocl_formosa_data_t *dd,
                                               _cl_command_node *node) {
  cl_event event = node->sync.event.event;
  cl_device_id dev = node->device;
  _cl_command_t *cmd = &node->command;
  size_t size = 0;

  switch (node->type) {
    case CL_COMMAND_READ_BUFFER:
      size = cmd->read.size;
      break;
    case CL_COMMAND_WRITE_BUFFER:
      size = cmd->write.size;
      break;
    case CL_COMMAND_COPY_BUFFER:
      size = cmd->copy.size;
      break;
    default:
      assert(0 && "not a Formosa memory-copy command");
      return;
  }

  pocl_update_event_running(event);
  formosa_pending_copy_t *pending =
      size == 0 ? NULL : (formosa_pending_copy_t *)calloc(1, sizeof(*pending));
  if (size != 0 && pending == NULL) {
    formosa_finish_command(event, CL_OUT_OF_HOST_MEMORY, NULL,
                           "Formosa Memory Copy");
    return;
  }

  FsaCompletionToken token = 0;
  cl_int err = CL_SUCCESS;
  switch (node->type) {
    case CL_COMMAND_READ_BUFFER:
      err = formosa_submit_read_buf(
          dd, cmd->read.dst_host_ptr,
          &POCL_MEM_BS(cmd->read.src)->device_ptrs[dev->global_mem_id],
          cmd->read.src, cmd->read.offset, size, &token);
      break;
    case CL_COMMAND_WRITE_BUFFER:
      err = formosa_submit_write_buf(
          dd, cmd->write.src_host_ptr,
          &POCL_MEM_BS(cmd->write.dst)->device_ptrs[dev->global_mem_id],
          cmd->write.dst, cmd->write.offset, size, &token);
      break;
    case CL_COMMAND_COPY_BUFFER:
      err = formosa_submit_copy_buf(
          dd, &POCL_MEM_BS(cmd->copy.dst)->device_ptrs[dev->global_mem_id],
          cmd->copy.dst,
          &POCL_MEM_BS(cmd->copy.src)->device_ptrs[dev->global_mem_id],
          cmd->copy.src, cmd->copy.dst_offset, cmd->copy.src_offset, size,
          &token);
      break;
  }

  if (err == CL_SUCCESS && size != 0) {
    pending->node = node;
    pending->token = token;
    formosa_enqueue_pending_copy(dd, pending);
    return;
  }

  free(pending);
  formosa_finish_command(event, err, "Event Memory Copy            ",
                         "Formosa Memory Copy");
}

static void formosa_command_scheduler(pocl_formosa_data_t *dd) {
  _cl_command_node *node;

  /* execute commands from ready list */
  while ((node = dd->ready_list)) {
    assert(pocl_command_is_ready(node->sync.event.event));
    assert(node->sync.event.event->status == CL_SUBMITTED);
    CDL_DELETE(dd->ready_list, node);
    POCL_UNLOCK(dd->cq_lock);

    cl_event event = node->sync.event.event;
    cl_int err = CL_SUCCESS;

    switch (node->type) {
      case CL_COMMAND_NDRANGE_KERNEL:
        pocl_update_event_running(event);
        err = formosa_run_kernel(dd, node);
        formosa_finish_command(event, err, "Event Enqueue NDRange       ",
                               "Formosa NDRange Kernel");
        break;

      case CL_COMMAND_READ_BUFFER:
      case CL_COMMAND_WRITE_BUFFER:
      case CL_COMMAND_COPY_BUFFER:
        formosa_submit_memory_copy_command(dd, node);
        break;

      case CL_COMMAND_READ_BUFFER_RECT:
      case CL_COMMAND_WRITE_BUFFER_RECT:
      case CL_COMMAND_COPY_BUFFER_RECT:
        /* ponytail: rectangular copies need grouped completions before they
         * can use the 1D firmware packet. */
        pocl_update_event_running(event);
        formosa_finish_command(event, CL_INVALID_OPERATION, NULL,
                               "Formosa rectangular buffer transfer");
        break;

      case CL_COMMAND_GRAPH_LAUNCH_FORMOSA:
        pocl_update_event_running(event);
        err = pocl_formosa_run_work_graph(dd, node);
        formosa_finish_command(event, err, "Event FSA Graph Launch      ",
                               "Formosa WorkGraph");
        break;

      default:
        pocl_exec_command(node);
        break;
    }

    POCL_LOCK(dd->cq_lock);
  }
}

static void *formosa_copy_completion_thread(void *arg) {
  pocl_formosa_data_t *dd = (pocl_formosa_data_t *)arg;
  while (1) {
    formosa_pending_copy_t *finished = NULL;
    FsaCompletionResult completion_result = FSA_COMPLETION_RESULT_PENDING;
    FsaCompletionPollStatus poll_status = kFsaCompletionPollPending;

    POCL_LOCK(dd->copy_lock);
    while (!dd->copy_thread_stop && dd->copy_pending == NULL)
      POCL_WAIT_COND(dd->copy_cond, dd->copy_lock);
    if (dd->copy_thread_stop) {
      POCL_UNLOCK(dd->copy_lock);
      break;
    }

    formosa_pending_copy_t *candidate = dd->copy_pending;
    POCL_UNLOCK(dd->copy_lock);

    if (candidate != NULL)
      poll_status = fsa_poll_completion(candidate->token, &completion_result);

    if (poll_status != kFsaCompletionPollPending) {
      POCL_LOCK(dd->copy_lock);
      if (dd->copy_pending == candidate) {
        dd->copy_pending = candidate->next;
        if (dd->copy_pending == NULL) dd->copy_pending_tail = NULL;
        finished = candidate;
      }
      POCL_UNLOCK(dd->copy_lock);
      if (poll_status == kFsaCompletionPollTransportError)
        formosa_mark_unavailable();
    }

    if (finished != NULL) {
      const cl_int event_status =
          poll_status == kFsaCompletionPollTerminal
              ? formosa_memory_copy_outcome_to_cl(completion_result)
              : CL_DEVICE_NOT_AVAILABLE;
      formosa_finish_command(finished->node->sync.event.event, event_status,
                             "Event Memory Copy            ",
                             "Formosa Memory Copy");
      if (poll_status == kFsaCompletionPollTerminal) {
        const FsaCompletionReleaseStatus release_status =
            fsa_release_completion(finished->token);
        if (release_status == kFsaCompletionReleaseTransportError)
          formosa_mark_unavailable();
        else if (release_status != kFsaCompletionReleaseAccepted)
          POCL_MSG_ERR("Formosa completion release failed (%d)\n",
                       release_status);
      }
      POCL_MEM_FREE(finished);

      POCL_LOCK(dd->cq_lock);
      formosa_command_scheduler(dd);
      POCL_UNLOCK(dd->cq_lock);
    } else {
      usleep(1000);
    }
  }
  return NULL;
}

void pocl_formosa_submit(_cl_command_node *node, cl_command_queue cq) {
  pocl_formosa_data_t *dd = node->device->data;

  node->state = POCL_COMMAND_READY;
  POCL_LOCK(dd->cq_lock);
  pocl_command_push(node, &dd->ready_list, &dd->command_list);

  POCL_UNLOCK_OBJ(node->sync.event.event);
  formosa_command_scheduler(dd);
  POCL_UNLOCK(dd->cq_lock);
}

void pocl_formosa_join(cl_device_id device, cl_command_queue cq) {
  pocl_formosa_data_t *dd = (pocl_formosa_data_t *)device->data;
  if (dd == NULL) return;

  while (CL_TRUE) {
    POCL_LOCK(dd->cq_lock);
    formosa_command_scheduler(dd);
    POCL_UNLOCK(dd->cq_lock);

    POCL_LOCK_OBJ(cq);
    const cl_bool done = cq->command_count == 0;
    POCL_UNLOCK_OBJ(cq);
    if (done) return;
    usleep(1000);
  }
}

void pocl_formosa_flush(cl_device_id device, cl_command_queue cq) {
  pocl_formosa_data_t *dd = (pocl_formosa_data_t *)device->data;
  if (dd == NULL) return;

  POCL_LOCK(dd->cq_lock);
  formosa_command_scheduler(dd);
  POCL_UNLOCK(dd->cq_lock);
}

void pocl_formosa_notify(cl_device_id device, cl_event event,
                         cl_event finished) {
  pocl_formosa_data_t *dd = (pocl_formosa_data_t *)device->data;
  if (dd == NULL) return;

  _cl_command_node *volatile node = event->command;

  if (finished->status < CL_COMPLETE) {
    /* Unlock the finished event in order to prevent a lock order violation
     * with the command queue that will be locked during
     * pocl_update_event_failed.
     */
    pocl_unlock_events_inorder(event, finished);
    pocl_update_event_failed(CL_FAILED, NULL, 0, event, NULL);
    /* Lock events in this order to avoid a lock order violation between
     * the finished/notifier and event/wait events.
     */
    pocl_lock_events_inorder(finished, event);
    return;
  }

  if (node->state != POCL_COMMAND_READY) {
    POCL_MSG_PRINT_EVENTS(
        "formosa: command related to the notified event %lu not ready\n",
        event->id);
    return;
  }

  if (pocl_command_is_ready(event)) {
    if (event->status == CL_QUEUED) {
      pocl_update_event_submitted(event);
      POCL_LOCK(dd->cq_lock);
      CDL_DELETE(dd->command_list, node);
      CDL_PREPEND(dd->ready_list, node);
      POCL_UNLOCK_OBJ(event);
      formosa_command_scheduler(dd);
      POCL_LOCK_OBJ(event);
      POCL_UNLOCK(dd->cq_lock);
    }
  }
}

/************************
 * Command Execution    *
 * **********************/

void pocl_formosa_read(void *data, void *__restrict__ host_ptr,
                       pocl_mem_identifier *src_mem_id, cl_mem src_buf,
                       size_t offset, size_t size) {
  cl_int err =
      formosa_read_buf(data, host_ptr, src_mem_id, src_buf, offset, size);
  if (err != CL_SUCCESS) POCL_MSG_ERR("pocl_formosa_read: failed (%d)\n", err);
}

void pocl_formosa_write(void *data, const void *__restrict__ host_ptr,
                        pocl_mem_identifier *dst_mem_id, cl_mem dst_buf,
                        size_t offset, size_t size) {
  cl_int err =
      formosa_write_buf(data, host_ptr, dst_mem_id, dst_buf, offset, size);
  if (err != CL_SUCCESS) POCL_MSG_ERR("pocl_formosa_write: failed (%d)\n", err);
}
