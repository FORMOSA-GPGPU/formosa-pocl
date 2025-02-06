#include "pocl-formosa.h"

#include <libcomm/comm.h>
#include <libcomm/msg.h>
#include <signal.h>
#include <stdlib.h>
#include <unistd.h>

#include "casvp-config/casvp-config.h"
#include "common.h"
#include "common_driver.h"
#include "falloc/fsa_mem_allocator.h"
#include "pocl-formosa-util.h"
#include "pocl_llvm.h"
#include "pocl_util.h"

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
  ops->copy = pocl_driver_copy;

  ops->get_mapping_ptr = pocl_driver_get_mapping_ptr;
  ops->free_mapping_ptr = pocl_driver_free_mapping_ptr;
}

/**************************
 * Probe & Initialization *
 **************************/

static cl_bool formosa_available = CL_TRUE;
static char *formosa_build_hash = "formosa-riscv64-unknown-unknwon-elf";

unsigned int pocl_formosa_probe(struct pocl_device_ops *ops) {
  struct timeval timeout;
  timeout.tv_sec = 0;
  timeout.tv_usec = 100000;  // 100ms
  int client_socket = client_connect(getenv("AGENT_SOCKET_PATH"), &timeout);
  if (client_socket == -1) {
    formosa_available = CL_FALSE;
    return 0;
  }
  int err = ipc_send_probe(client_socket);
  if (err == -1) {
    formosa_available = CL_FALSE;
    return 0;
  }
  close(client_socket);
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
  device->device_side_printf = 0;
  device->has_64bit_long = CL_TRUE;

  device->address_bits = 64;
  device->llvm_target_triplet = "riscv64-unknown-unknown-elf";
  device->llvm_abi = "lp64";
  device->llvm_cpu = "formosa-gpgpu";
  device->kernellib_name = "kernel-riscv64-formosa";
  device->kernellib_fallback_name = NULL;
  device->kernellib_subdir = "formosa";

  device->image_support = CL_FALSE;

  size_t num_warps = CASVP_FORMOSA_WARPS_PER_CORE;
  size_t num_threads = CASVP_FORMOSA_THREADS_PER_WARP;
  uint64_t max_work_group_size = num_warps * num_threads;

  device->global_mem_cache_type = CL_READ_WRITE_CACHE;
  device->global_mem_cacheline_size = CASVP_FORMOSA_CACHE_BLOCK_SIZE;
  device->global_mem_cache_size = CASVP_FORMOSA_CACHE_SIZE;
  device->global_mem_size = CASVP_FORMOSA_GLOBAL_MEM_SIZE;
  device->max_mem_alloc_size = CASVP_FORMOSA_GLOBAL_MEM_SIZE;
  device->local_mem_size = CASVP_FORMOSA_LOCAL_MEM_SIZE;
  device->max_work_group_size = max_work_group_size;
  device->max_work_item_sizes[0] = max_work_group_size;
  device->max_work_item_sizes[1] = max_work_group_size;
  device->max_work_item_sizes[2] = max_work_group_size;
  device->max_compute_units = 1;
  device->mem_base_addr_align = 128;  // TODO: determine this

  dd->context_ref_count = 0;

  POCL_INIT_LOCK(dd->compile_lock);
  POCL_INIT_LOCK(dd->cq_lock);

  device->data = dd;
  device->available = &formosa_available;

  // Connect to virtual platform
  dd->client_fd = client_connect(getenv("AGENT_SOCKET_PATH"), NULL);
  if (dd->client_fd == -1) {
    formosa_available = CL_FALSE;
    return CL_DEVICE_NOT_FOUND;
  }
  // Register interrupt signal
  signal(SIGUSR1, fsa_int_handler);
  if (ipc_register_signal(dd->client_fd, SIGUSR1) == -1) {
    formosa_available = CL_FALSE;
    return CL_DEVICE_NOT_FOUND;
  }
  dd->kernel_buffer = NULL;
  formosa_available = CL_TRUE;

  fsaMemAllocInit(FSA_GLOBAL_MEM_BASE, CASVP_FORMOSA_GLOBAL_MEM_SIZE, 0);

  return CL_SUCCESS;
}

cl_int pocl_formosa_uninit(unsigned j, cl_device_id device) {
  pocl_formosa_data_t *dd = (pocl_formosa_data_t *)device->data;
  if (dd == NULL) return CL_SUCCESS;

  POCL_DESTROY_LOCK(dd->compile_lock);
  POCL_DESTROY_LOCK(dd->cq_lock);
  if (dd->client_fd != -1) {
    close(dd->client_fd);
    dd->client_fd = -1;
  }
  if (dd->kernel_buffer != NULL) {
    POCL_MEM_FREE(dd->kernel_buffer);
  }
  POCL_MEM_FREE(device->data);
  device->data = NULL;
  return CL_SUCCESS;
}

void pocl_formosa_run(void *data, _cl_command_node *cmd) {
  pocl_formosa_data_t *dd;
  struct pocl_argument *al;
  cl_uint device_i = cmd->program_device_i;
  cl_kernel kernel = cmd->command.run.kernel;
  cl_program program = kernel->program;
  pocl_kernel_metadata_t *meta = kernel->meta;
  formosa_program_data_t *pdata =
      (formosa_program_data_t *)program->data[device_i];
  formosa_kernel_data_t *kdata = (formosa_kernel_data_t *)meta->data[device_i];
  struct pocl_context *pc = &cmd->command.run.pc;
  int err = 0;

  uint32_t num_groups = 1;
  uint32_t group_size = 1;
  for (uint32_t i = 0; i < pc->work_dim; ++i) {
    num_groups *= pc->num_groups[i];
    group_size *= pc->local_size[i];
  }
  if (num_groups == 0 || group_size == 0) return;

  assert(data != NULL);
  dd = (pocl_formosa_data_t *)data;

  const uint32_t ptr_size = 8;
  const uint32_t word_size = 8;

  // calculate kernel arguments buffer size
  uint32_t local_mem_size = 0;   // total local memory size
  size_t kargs_buffer_size = 0;  // kernel argument buffer size

  for (int i = 0; i < meta->num_args; ++i) {
    struct pocl_argument *al = &(cmd->command.run.arguments[i]);
    if (ARG_IS_LOCAL(meta->arg_info[i])) {
      local_mem_size += al->size;
      kargs_buffer_size += word_size;
    } else if ((meta->arg_info[i].type == POCL_ARG_TYPE_POINTER) ||
               (meta->arg_info[i].type == POCL_ARG_TYPE_IMAGE) ||
               (meta->arg_info[i].type == POCL_ARG_TYPE_SAMPLER)) {
      kargs_buffer_size += ptr_size;
    } else {
      // scalar argument
      kargs_buffer_size += al->size;
      if (al->size % word_size != 0) {
        kargs_buffer_size += word_size - (al->size % word_size);
      }
    }
  }

  // local buffers
  for (int i = 0; i < meta->num_locals; ++i) {
    local_mem_size += meta->local_sizes[i];
    kargs_buffer_size += word_size;
  }

  // add local size
  if (local_mem_size != 0) {
    kargs_buffer_size += word_size;
  }

  // check occupancy
  if (group_size != 1) {
    uint32_t available_local_mem;
    err = fsa_check_occupancy(group_size, &available_local_mem);
    if (err != 0) {
      POCL_ABORT("ERROR (pocl_formosa_run): Check occupancy failed\n");
    }
    if (local_mem_size > available_local_mem) {
      POCL_ABORT(
          "ERROR (pocl_formosa_run): Out of local memory: needed=%d bytes, "
          "available=%d bytes\n",
          local_mem_size, available_local_mem);
    }
  }

  // allocate arguments host buffer
  uint8_t *const host_kargs_base_ptr = malloc(kargs_buffer_size);
  assert(host_kargs_base_ptr);

  // allocate kernel arguments buffer
  formosa_buffer_data_t fsa_kargs_buffer;
  memset(&fsa_kargs_buffer, 0, sizeof(formosa_buffer_data_t));
  void *device_args_buffer_addr;
  err = fsaMalloc(&device_args_buffer_addr, kargs_buffer_size);
  if (err != 0) {
    POCL_ABORT("ERROR (pocl_formosa_run): Device memory allocation failed\n");
  }
  fsa_kargs_buffer.buf_address = (uint64_t)device_args_buffer_addr;
  fsa_kargs_buffer.buf_size = kargs_buffer_size;
  fsa_kargs_buffer.client_fd = dd->client_fd;

  // write context data
  fsa_write_csr(dd, CASVP_FORMOSA_CSR_DIM, pc->work_dim);
  fsa_write_csr(dd, CASVP_FORMOSA_CSR_LAUNCH_KERNEL_ID, kdata->kernel_id);
  fsa_write_csr(dd, CASVP_FORMOSA_CSR_KERNEL_ARG,
                (uint64_t)device_args_buffer_addr);
  FSA_WRITE_GROUPED_CSR(dd, CASVP_FORMOSA_CSR_LOCAL_SIZE, pc->local_size);
  FSA_WRITE_GROUPED_CSR(dd, CASVP_FORMOSA_CSR_NUM_GROUPS, pc->num_groups);

  // write arguments
  uint32_t host_args_offset = 0;
  uint32_t local_mem_offset = 0;

  for (int i = 0; i < meta->num_args; ++i) {
    struct pocl_argument *al = &(cmd->command.run.arguments[i]);
    if (ARG_IS_LOCAL(meta->arg_info[i])) {
      if (local_mem_offset == 0) {
        memcpy(host_kargs_base_ptr + host_args_offset, &local_mem_size,
               4);  // total local memory size
        host_args_offset += word_size;
      }
      memcpy(host_kargs_base_ptr + host_args_offset, &local_mem_offset,
             4);  // local memory offset
      host_args_offset += word_size;
      local_mem_offset += al->size;
    } else if (meta->arg_info[i].type == POCL_ARG_TYPE_POINTER) {
      if (al->value == NULL) {
        memset(host_kargs_base_ptr + host_args_offset, 0,
               ptr_size);  // NULL pointer value
        host_args_offset += ptr_size;
      } else {
        cl_mem m = (*(cl_mem *)(al->value));
        formosa_buffer_data_t *buf_data =
            (formosa_buffer_data_t *)m->device_ptrs[cmd->device->global_mem_id]
                .mem_ptr;
        uint64_t dev_mem_addr = buf_data->buf_address + al->offset;
        memcpy(host_kargs_base_ptr + host_args_offset, &dev_mem_addr,
               ptr_size);  // pointer value
        host_args_offset += ptr_size;
      }
    } else if (meta->arg_info[i].type == POCL_ARG_TYPE_IMAGE) {
      POCL_ABORT("ERROR (pocl_formosa_run): Image argument not supported\n");
    } else if (meta->arg_info[i].type == POCL_ARG_TYPE_SAMPLER) {
      POCL_ABORT("ERROR (pocl_formosa_run): Sampler argument not supported\n");
    } else {
      // scalar argument
      memcpy(host_kargs_base_ptr + host_args_offset, al->value,
             al->size);  // scalar value
      host_args_offset += al->size;
      if (al->size % word_size != 0) {
        host_args_offset += word_size - (al->size % word_size);
      }
    }
  }

  // write local memory size
  for (int i = 0; i < meta->num_locals; ++i) {
    if (local_mem_offset == 0) {
      memcpy(host_kargs_base_ptr + host_args_offset, &local_mem_size,
             4);  // local_size
      host_args_offset += word_size;
    }
    memcpy(host_kargs_base_ptr + host_args_offset, &local_mem_offset,
           4);  // arg offset
    host_args_offset += word_size;
    local_mem_offset += meta->local_sizes[i];
  }

  // upload kernel arguments buffer
  err = fsa_copy_to_dev(&fsa_kargs_buffer, host_kargs_base_ptr, 0,
                        kargs_buffer_size);
  if (err != 0) {
    POCL_ABORT(
        "ERROR (pocl_formosa_run): Kernel argument copy to device failed\n");
  }

  // release argument host buffer
  free(host_kargs_base_ptr);

  // upload kernel to device
  if (dd->kernel_buffer == NULL) {
    char sz_program_fsabin[POCL_MAX_PATHNAME_LENGTH];
    err = fsa_get_elf_name(program, device_i, sz_program_fsabin);
    err |= fsa_upload_kernel_sections(sz_program_fsabin, dd);
    if (err != 0) {
      POCL_ABORT("ERROR (pocl_formosa_run): Kernel upload failed\n");
    }
    char *trampoline_name = malloc(strlen(kernel->name) + 12);
    sprintf(trampoline_name, "%s_trampoline", kernel->name);
    uint64_t kernel_pc = fsa_get_symbol_pc(sz_program_fsabin, trampoline_name);
    POCL_MEM_FREE(trampoline_name);
    uint64_t entry_pc = fsa_get_symbol_pc(sz_program_fsabin, "_start");
    err = fsa_write_csr(dd, CASVP_FORMOSA_CSR_KERNEL_PC, kernel_pc);
    err |= fsa_write_csr(dd, CASVP_FORMOSA_CSR_ENTRY_PC, entry_pc);
    if (err != 0) {
      POCL_ABORT("ERROR (pocl_formosa_run): Kernel CSR setup failed\n");
    }
  }

  // launch kernel execution
  uint64_t start_status = 0;
  do {
    err = fsa_read_csr(dd, CASVP_FORMOSA_CSR_START, &start_status);
  } while (start_status != 0);
  err = fsa_write_csr(dd, CASVP_FORMOSA_CSR_START, 1);
  if (err != 0) {
    POCL_ABORT("ERROR (pocl_formosa_run): Kernel launch failed\n");
  }

  // wait for the execution to complete
  err = fsa_wait_ack(dd);
  if (err != 0) {
    POCL_ABORT("ERROR (pocl_formosa_run): Kernel execution failed\n");
  }

  // release arguments device buffer
  err = fsaFree((void *)fsa_kargs_buffer.buf_address);
  if (err != 0) {
    POCL_ABORT("ERROR (pocl_formosa_run): Kernel argument free failed\n");
  }
}

/**************************
 * Program                *
 **************************/

char *pocl_formosa_init_build(void *data) {
  // clang -cc1 options
  return strdup("");
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
  err = fsa_get_elf_name(program, device_i, fsa_program_bin);
  if (err != 0) {
    POCL_MSG_ERR("Get ELF name failed\n");
    goto POST_BUILD_PROGRAM_FINALLY;
  }
  err = fsa_compile_program(&pdata->kernel_names, &pdata->num_kernels,
                            fsa_program_bin, program->llvm_irs[device_i]);

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
  kdata->kernel_id = i;
  ++kdata->ref_count;

  meta->data[program_device_i] = kdata;

  return CL_SUCCESS;
}

cl_int pocl_formosa_free_kernel(cl_device_id device, cl_program program,
                                cl_kernel kernel, unsigned program_device_i) {
  pocl_kernel_metadata_t *meta = kernel->meta;
  assert(meta->data != NULL);
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
  int err = fsaMalloc(&addr, mem_obj->size);
  if (err) {
    return CL_MEM_OBJECT_ALLOCATION_FAILURE;
  }

  formosa_buffer_data_t *temp = malloc(sizeof(formosa_buffer_data_t));
  if (temp == NULL) {
    fsaFree((void *)addr);
    return CL_OUT_OF_HOST_MEMORY;
  }
  memset(temp, 0, sizeof(formosa_buffer_data_t));
  pocl_formosa_data_t *dd = (pocl_formosa_data_t *)device->data;
  if (host_ptr) {  // READ_WRITE, WRITE_ONLY
    temp->buf_address = (uint64_t)addr;
    temp->buf_size = mem_obj->size;
    temp->client_fd = dd->client_fd;
    temp->msg_id = 0;
    err = fsa_copy_to_dev(temp, host_ptr, 0, mem_obj->size);
    if (err != 0) {
      fsaFree((void *)temp->buf_address);
      POCL_MEM_FREE(temp);
      return CL_OUT_OF_RESOURCES;
    }
  } else {  // READ_ONLY
    temp->buf_address = (uint64_t)addr;
    temp->buf_size = mem_obj->size;
    temp->client_fd = dd->client_fd;
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
    POCL_ABORT("ERROR (pocl_formosa_free): Memory flag not supported\n");
  }
  if (flags & CL_MEM_ALLOC_HOST_PTR) {
    pocl_release_mem_host_ptr(mem_obj);
  }
  fsaFree((void *)fb->buf_address);
  POCL_MEM_FREE(fb);
  p->mem_ptr = NULL;
}

/**************************
 * Event Handling         *
 **************************/

static void formosa_command_scheduler(pocl_formosa_data_t *dd) {
  _cl_command_node *node;

  /* execute commands from ready list */
  while ((node = dd->ready_list)) {
    assert(pocl_command_is_ready(node->sync.event.event));
    assert(node->sync.event.event->status == CL_SUBMITTED);
    CDL_DELETE(dd->ready_list, node);
    POCL_UNLOCK(dd->cq_lock);
    pocl_exec_command(node);
    POCL_LOCK(dd->cq_lock);
  }
}

void pocl_formosa_submit(_cl_command_node *node, cl_command_queue cq) {
  pocl_formosa_data_t *dd = node->device->data;

  node->ready = 1;
  POCL_LOCK(dd->cq_lock);
  pocl_command_push(node, &dd->ready_list, &dd->command_list);

  POCL_UNLOCK_OBJ(node->sync.event.event);
  formosa_command_scheduler(dd);
  POCL_UNLOCK(dd->cq_lock);
}

void pocl_formosa_join(cl_device_id device, cl_command_queue cq) {
  pocl_formosa_data_t *dd = (pocl_formosa_data_t *)device->data;
  if (dd == NULL) return;

  POCL_LOCK(dd->cq_lock);
  formosa_command_scheduler(dd);
  POCL_UNLOCK(dd->cq_lock);
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
    pocl_update_event_failed(event);
    return;
  }

  if (!node->ready) return;

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
  formosa_buffer_data_t *buffer_data =
      (formosa_buffer_data_t *)src_mem_id->mem_ptr;
  int err = fsa_copy_from_dev(buffer_data, host_ptr, offset, size);
  if (err != 0) {
    POCL_ABORT("ERROR (pocl_formosa_read): Copy from device failed\n");
  }
}

void pocl_formosa_write(void *data, const void *__restrict__ host_ptr,
                        pocl_mem_identifier *dst_mem_id, cl_mem dst_buf,
                        size_t offset, size_t size) {
  formosa_buffer_data_t *buffer_data =
      (formosa_buffer_data_t *)dst_mem_id->mem_ptr;
  int err = fsa_copy_to_dev(buffer_data, host_ptr, offset, size);
  if (err != 0) {
    POCL_ABORT("ERROR (pocl_formosa_write): Copy to device failed\n");
  }
}
