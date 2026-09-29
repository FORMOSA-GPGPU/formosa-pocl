#include "pocl-simtsim.h"

#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "common.h"
#include "common_driver.h"
#include "falloc.h"
#include "pocl_cl.h"
#include "pocl_llvm.h"
#include "pocl_util.h"
#include "simtsim-util.h"
#include "simtsim.h"

static cl_bool simtsim_available = CL_TRUE;
static char *simtsim_build_hash = "formosa-riscv64-unknown-unknwon-elf";

void pocl_simtsim_init_device_ops(struct pocl_device_ops *ops) {
  ops->device_name = "simtsim";
  ops->build_hash = pocl_simtsim_build_hash;
  ops->probe = pocl_simtsim_probe;
  ops->init = pocl_simtsim_init;
  ops->uninit = pocl_simtsim_uninit;

  ops->init_context = pocl_simtsim_init_context;
  ops->free_context = pocl_simtsim_free_context;

  ops->run = pocl_simtsim_run;
  ops->run_native = NULL;

  ops->alloc_mem_obj = pocl_simtsim_alloc_mem_obj;
  ops->free = pocl_simtsim_free;

  ops->init_build = pocl_simtsim_init_build;
  ops->build_source = pocl_driver_build_source;
  ops->link_program = pocl_driver_link_program;
  ops->build_binary = pocl_driver_build_binary;
  ops->setup_metadata = pocl_driver_setup_metadata;
  ops->supports_binary = pocl_driver_supports_binary;
  ops->build_poclbinary = pocl_driver_build_poclbinary;
  ops->build_builtin = pocl_driver_build_opencl_builtins;

  ops->post_build_program = pocl_simtsim_post_build_program;
  ops->free_program = pocl_simtsim_free_program;

  ops->create_kernel = pocl_simtsim_create_kernel;
  ops->free_kernel = pocl_simtsim_free_kernel;

  ops->submit = pocl_simtsim_submit;
  ops->join = pocl_simtsim_join;
  ops->flush = pocl_simtsim_flush;
  ops->notify = pocl_simtsim_notify;
  ops->broadcast = pocl_broadcast;

  ops->read = pocl_simtsim_read;
  ops->write = pocl_simtsim_write;
  ops->copy = pocl_driver_copy;

  ops->get_mapping_ptr = pocl_driver_get_mapping_ptr;
  ops->free_mapping_ptr = pocl_driver_free_mapping_ptr;

  ops->map_mem = pocl_simtsim_map_mem;
  ops->unmap_mem = pocl_simtsim_unmap_mem;
}

char *pocl_simtsim_build_hash(cl_device_id device) {
  char *res = calloc(strlen(simtsim_build_hash) + 1, sizeof(char));
  strncpy(res, simtsim_build_hash, strlen(simtsim_build_hash));
  return res;
}

unsigned int pocl_simtsim_probe(struct pocl_device_ops *ops) {
  return strncmp(ops->device_name, "simtsim", 7) == 0;
}

static void parse_simtsim_env(SimtsimConfig *config) {
  memset(config, 0, sizeof(SimtsimConfig));

  char *env_val = getenv("SIMTSIM_CONFIG");
  if (!env_val)
    return;

  char *env_copy = strdup(env_val);
  if (!env_copy)
    return;

  char *token = strtok(env_copy, ",");
  while (token != NULL) {
    if (strcmp(token, "debug") == 0)
      config->debug = true;
    else if (strcmp(token, "mt") == 0)
      config->mt = true;
    else if (strcmp(token, "mt2") == 0)
      config->mt2 = true;
    else if (strcmp(token, "simd") == 0)
      config->simd = true;
    else if (strcmp(token, "ct") == 0)
      config->ct = true;
    else if (strcmp(token, "aot") == 0)
      config->aot = true;
    else if (strcmp(token, "cuda") == 0)
      config->cuda = true;
    else if (strcmp(token, "bench") == 0)
      config->bench = true;
    else if (strcmp(token, "sremap") == 0)
      config->sremap = true;
    else if (sscanf(token, "thread=%d", &config->num_threads) == 1)
      ;
    else if (sscanf(token, "maxreg=%d", &config->maxreg) == 1)
      ;
    else if (strncmp(token, "ptx=", 4) == 0)
      config->ptx_file = strdup(token + 4);

    token = strtok(NULL, ",");
  }

  /* Set default thread count if not specified */
  if (config->num_threads == 0)
    config->num_threads = sysconf(_SC_NPROCESSORS_ONLN);

  free(env_copy);
}

static void init_device_memory(SimtsimSystem *sim, uint64_t mem_size) {
  fsa_init(0, mem_size, NULL);
  /* Region for code 4MB */
  fsa_addr_malloc(0, 0x400000);
}

cl_int pocl_simtsim_init(unsigned j, cl_device_id device,
                         const char *parameters) {
  pocl_simtsim_data_t *dd;
  assert(device->data == NULL);
  pocl_init_default_device_infos(device, SIMTSIM_DEVICE_EXTENSIONS);

  SETUP_DEVICE_CL_VERSION(device, SIMTSIM_DEVICE_CL_VERSION_MAJOR,
                          SIMTSIM_DEVICE_CL_VERSION_MINOR);

  dd = (pocl_simtsim_data_t *)calloc(1, sizeof(pocl_simtsim_data_t));
  if (dd == NULL) {
    return CL_OUT_OF_HOST_MEMORY;
  }
  SimtsimConfig config;
  parse_simtsim_env(&config);
  char device_long_name[128] = "SIMTSim Formosa GPGPU (";
  char *config_str = getenv("SIMTSIM_CONFIG");
  if (config_str) {
    strcat(device_long_name, getenv("SIMTSIM_CONFIG"));
  }
  strcat(device_long_name, ")");

  device->vendor = "CASLab";
  device->long_name = strdup(device_long_name);
  device->short_name = "simtsim";
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
  device->kernellib_name = "kernel-riscv64-simtsim";
  device->kernellib_fallback_name = NULL;
  device->kernellib_subdir = "simtsim";

  device->image_support = CL_FALSE;

  /* TODO: Find out how these fields affect the runtime */
  size_t num_warps = 1024;
  size_t num_threads = 32;
  uint64_t max_work_group_size = num_warps * num_threads;

  device->global_mem_cache_type = CL_READ_WRITE_CACHE;
  device->global_mem_cacheline_size = 128;
  uint64_t mem_size = 0x100000000;
  char *mem_size_str = getenv("SIMTSIM_MEM_SIZE");
  if (mem_size_str) {
    uint64_t requested = strtoull(mem_size_str, NULL, 0);
    if (requested > 0) {
      mem_size = requested;
    }
  }
  device->global_mem_cache_size = mem_size;
  device->global_mem_size = mem_size;
  device->max_mem_alloc_size = mem_size;
  device->local_mem_size = mem_size;
  device->max_work_group_size = max_work_group_size;
  device->max_work_item_sizes[0] = max_work_group_size;
  device->max_work_item_sizes[1] = max_work_group_size;
  device->max_work_item_sizes[2] = max_work_group_size;
  device->max_compute_units = 1;
  device->mem_base_addr_align = 128;

  dd->ref_count = 0;
  POCL_INIT_LOCK(dd->compile_lock);
  POCL_INIT_LOCK(dd->cq_lock);

  device->data = dd;
  device->available = &simtsim_available;

  dd->sim = simtsim_create(mem_size, config);
  init_device_memory(dd->sim, mem_size);

  dd->kernel_buffer = NULL;
  simtsim_available = CL_TRUE;

  if (!dd->sim) {
    simtsim_available = CL_FALSE;
    POCL_ABORT("ERROR (pocl_simtsim_init): Driver initialization failed\n");
  }

  return CL_SUCCESS;
}

cl_int pocl_simtsim_uninit(unsigned j, cl_device_id device) {
  pocl_simtsim_data_t *dd = (pocl_simtsim_data_t *)device->data;
  if (dd == NULL) {
    return CL_SUCCESS;
  }
  POCL_DESTROY_LOCK(dd->compile_lock);
  POCL_DESTROY_LOCK(dd->cq_lock);
  if (dd->kernel_buffer != NULL) {
    POCL_MEM_FREE(dd->kernel_buffer);
  }
  simtsim_destroy(dd->sim);
  fsa_clean();
  dd->sim = NULL;
  POCL_MEM_FREE(device->data);
  device->data = NULL;
  return CL_SUCCESS;
}

cl_int pocl_simtsim_init_context(cl_device_id device, cl_context context) {
  pocl_simtsim_data_t *dd = (pocl_simtsim_data_t *)device->data;
  if (!dd) {
    return CL_FAILED;
  }
  return CL_SUCCESS;
}

cl_int pocl_simtsim_free_context(cl_device_id device, cl_context context) {
  pocl_simtsim_data_t *dd = (pocl_simtsim_data_t *)device->data;
  if (!dd) {
    return CL_SUCCESS;
  }

  dd->ref_count--;
  if (dd->ref_count == 0) {
    pocl_simtsim_uninit(0, device);
  }
  return CL_SUCCESS;
}

char *pocl_simtsim_init_build(void *data) { return strdup(""); }

int pocl_simtsim_post_build_program(cl_program program, cl_uint device_i) {
  cl_device_id dev = program->devices[device_i];
  pocl_simtsim_data_t *ddata = (pocl_simtsim_data_t *)dev->data;
  simtsim_program_data_t *pdata = NULL;

  POCL_LOCK(ddata->compile_lock);
  int err = pocl_llvm_run_passes_on_program(program, device_i);
  if (err != CL_SUCCESS) {
    POCL_MSG_ERR("LLVM passes failed for program\n");
    goto POST_BUILD_PROGRAM_FINALLY;
  }

  pdata = calloc(1, sizeof(*pdata));
  pdata->names = NULL;

  char program_bin[POCL_MAX_PATHNAME_LENGTH];
  err = pocl_simtsim_get_elf_name(program, device_i, program_bin);
  if (err != 0) {
    POCL_MSG_ERR("Get ELF name failed\n");
    goto POST_BUILD_PROGRAM_FINALLY;
  }
  err = pocl_simtsim_compile_program(&pdata->names, &pdata->num_kernels,
                                     program_bin, program->compiler_options,
                                     program->llvm_irs[device_i]);

  /* Compile to simtsim program */
  SimtsimConfig cfg = simtsim_get_config(ddata->sim);
  pdata->sim_program =
      simtsim_create_program(program_bin, cfg.ct, cfg.aot, cfg.simd, cfg.cuda,
                             cfg.bench, cfg.sremap, 0);
  if (!pdata->sim_program) {
    err = CL_BUILD_PROGRAM_FAILURE;
  }
POST_BUILD_PROGRAM_FINALLY:
  program->data[device_i] = pdata;
  POCL_UNLOCK(ddata->compile_lock);
  return err;
}

int pocl_simtsim_free_program(cl_device_id device, cl_program program,
                              unsigned program_device_i) {
  pocl_simtsim_data_t *dd = (pocl_simtsim_data_t *)device->data;
  simtsim_program_data_t *pdata =
      (simtsim_program_data_t *)program->data[program_device_i];
  if (!pdata)
    return CL_SUCCESS;
  simtsim_program_destroy(pdata->sim_program);
  pocl_driver_free_program(device, program, program_device_i);
  POCL_MEM_FREE(pdata->names);
  POCL_MEM_FREE(pdata);
  program->data[program_device_i] = NULL;
  return CL_SUCCESS;
}

cl_int pocl_simtsim_create_kernel(cl_device_id device, cl_program program,
                                  cl_kernel kernel, unsigned program_device_i) {
  pocl_kernel_metadata_t *meta = kernel->meta;
  assert(meta->data != NULL);
  simtsim_kernel_data_t *kdata =
      (simtsim_kernel_data_t *)program->data[program_device_i];
  if (!kdata) {
    ++kdata->ref_count;
    return CL_SUCCESS;
  }

  simtsim_program_data_t *pdata =
      (simtsim_program_data_t *)program->data[program_device_i];
  assert(pdata != NULL);

  const char *current = pdata->names;
  int i = 0, found = 0;
  for (; i < pdata->num_kernels; ++i) {
    if (strcmp(current, kernel->name) == 0) {
      found = 1;
      break;
    }
    current += strlen(current) + 1;
  }
  assert(found);
  kdata = calloc(1, sizeof(simtsim_kernel_data_t));
  kdata->id = i;
  ++kdata->ref_count;
  meta->data[program_device_i] = kdata;
  return CL_SUCCESS;
}

cl_int pocl_simtsim_free_kernel(cl_device_id device, cl_program program,
                                cl_kernel kernel, unsigned program_device_i) {
  pocl_kernel_metadata_t *meta = kernel->meta;
  assert(meta->data != NULL);
  simtsim_kernel_data_t *kdata =
      (simtsim_kernel_data_t *)meta->data[program_device_i];
  if (!kdata) {
    return CL_SUCCESS;
  }

  if (--kdata->ref_count == 0) {
    POCL_MEM_FREE(kdata);
    meta->data[program_device_i] = NULL;
  }
  return CL_SUCCESS;
}

cl_int pocl_simtsim_alloc_mem_obj(cl_device_id device, cl_mem mem_obj,
                                  void *host_ptr) {
  pocl_simtsim_data_t *dd = (pocl_simtsim_data_t *)device->data;
  SimtsimSystem *sim = dd->sim;
  pocl_mem_identifier *p = &mem_obj->device_ptrs[device->global_mem_id];

  cl_mem_flags flags = mem_obj->flags;
  assert((flags & (CL_MEM_READ_WRITE | CL_MEM_WRITE_ONLY | CL_MEM_READ_ONLY)) !=
         0);

  void *addr;
  int err = fsa_malloc(&addr, mem_obj->size);
  if (err) {
    return CL_MEM_OBJECT_ALLOCATION_FAILURE;
  }

  simtsim_buffer_data_t *temp = malloc(sizeof(*temp));
  if (!temp) {
    fsa_free((void *)addr);
    return CL_OUT_OF_HOST_MEMORY;
  }
  memset(temp, 0, sizeof(*temp));
  if (host_ptr) {
    temp->addr = (uint64_t)addr;
    temp->size = mem_obj->size;
    simtsim_write_mem(sim, temp->addr, temp->size, host_ptr);
  } else {
    temp->addr = (uint64_t)addr;
    temp->size = mem_obj->size;
  }

  p->mem_ptr = temp;

  return CL_SUCCESS;
}

void pocl_simtsim_free(cl_device_id device, cl_mem mem_obj) {
  pocl_mem_identifier *p = &mem_obj->device_ptrs[device->global_mem_id];
  cl_mem_flags flags = mem_obj->flags;
  simtsim_buffer_data_t *b = (simtsim_buffer_data_t *)p->mem_ptr;
  if (!b) {
    POCL_ABORT("ERROR (pocl_simtsim_free): Memory flag not supported\n");
  }

  if (flags & CL_MEM_ALLOC_HOST_PTR) {
    pocl_release_mem_host_ptr(mem_obj);
  }
  fsa_free((void *)b->addr);
  POCL_MEM_FREE(b);
  p->mem_ptr = NULL;
}

static void simtsim_command_scheduler(pocl_simtsim_data_t *d) {
  _cl_command_node *node;
  while ((node = d->ready_list)) {
    assert(pocl_command_is_ready(node->sync.event.event));
    assert(node->sync.event.event->status == CL_SUBMITTED);
    CDL_DELETE(d->ready_list, node);
    POCL_UNLOCK(d->cq_lock);
    pocl_exec_command(node);
    POCL_LOCK(d->cq_lock);
  }
}

void pocl_simtsim_submit(_cl_command_node *node, cl_command_queue cq) {
  pocl_simtsim_data_t *dd = (pocl_simtsim_data_t *)node->device->data;

  node->state = POCL_COMMAND_READY;
  POCL_LOCK(dd->cq_lock);
  pocl_command_push(node, &dd->ready_list, &dd->command_list);

  POCL_UNLOCK_OBJ(node->sync.event.event);
  simtsim_command_scheduler(dd);
  POCL_UNLOCK(dd->cq_lock);
}

void pocl_simtsim_join(cl_device_id device, cl_command_queue cq) {
  pocl_simtsim_data_t *dd = (pocl_simtsim_data_t *)device->data;
  if (dd == NULL)
    return;

  POCL_LOCK(dd->cq_lock);
  simtsim_command_scheduler(dd);
  POCL_UNLOCK(dd->cq_lock);
}

void pocl_simtsim_flush(cl_device_id device, cl_command_queue cq) {
  pocl_simtsim_data_t *dd = (pocl_simtsim_data_t *)device->data;
  if (dd == NULL)
    return;

  POCL_LOCK(dd->cq_lock);
  simtsim_command_scheduler(dd);
  POCL_UNLOCK(dd->cq_lock);
}

void pocl_simtsim_notify(cl_device_id device, cl_event event,
                         cl_event finished) {
  pocl_simtsim_data_t *dd = (pocl_simtsim_data_t *)device->data;
  if (dd == NULL)
    return;

  _cl_command_node *volatile node = event->command;

  if (finished->status < CL_COMPLETE) {
    pocl_unlock_events_inorder(event, finished);
    pocl_update_event_failed(CL_FAILED, NULL, 0, event, NULL);
    pocl_lock_events_inorder(finished, event);
    return;
  }

  if (node->state != POCL_COMMAND_READY) {
    POCL_MSG_PRINT_EVENTS(
        "simtsim: command related to the notified event %" PRIu64
        " not ready\n",
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
      simtsim_command_scheduler(dd);
      POCL_LOCK_OBJ(event);
      POCL_UNLOCK(dd->cq_lock);
    }
  }
}

void pocl_simtsim_read(void *data, void *__restrict__ host_ptr,
                       pocl_mem_identifier *src_mem_id, cl_mem src_buf,
                       size_t offset, size_t size) {
  pocl_simtsim_data_t *dd = (pocl_simtsim_data_t *)data;
  SimtsimSystem *sim = dd->sim;
  simtsim_buffer_data_t *buffer_data =
      (simtsim_buffer_data_t *)src_mem_id->mem_ptr;
  if (buffer_data == NULL) {
    POCL_ABORT("ERROR (pocl_simtsim_read): Memory buffer not found\n");
  }
  if (offset + size > buffer_data->size) {
    POCL_ABORT("ERROR (pocl_simtsim_read): Out of buffer size\n");
  }
  int err = simtsim_read_mem(sim, host_ptr, buffer_data->addr + offset, size);
  if (err != 0) {
    POCL_ABORT("ERROR (pocl_simtsim_read): Copy from device failed\n");
  }
}

void pocl_simtsim_write(void *data, const void *__restrict__ host_ptr,
                        pocl_mem_identifier *dst_mem_id, cl_mem dst_buf,
                        size_t offset, size_t size) {
  pocl_simtsim_data_t *dd = (pocl_simtsim_data_t *)data;
  SimtsimSystem *sim = dd->sim;
  simtsim_buffer_data_t *buffer_data =
      (simtsim_buffer_data_t *)dst_mem_id->mem_ptr;
  if (buffer_data == NULL) {
    POCL_ABORT("ERROR (pocl_simtsim_write): Memory buffer not found\n");
  }
  if (offset + size > buffer_data->size) {
    POCL_ABORT("ERROR (pocl_simtsim_write): Out of buffer size\n");
  }

  int err = simtsim_write_mem(sim, buffer_data->addr + offset, size,
                              (uint8_t *)host_ptr);
  if (err != 0) {
    POCL_ABORT("ERROR (pocl_simtsim_write): Copy to device failed\n");
  }
}

cl_int pocl_simtsim_map_mem(void *data, pocl_mem_identifier *src_mem_id,
                            cl_mem src_buf, mem_mapping_t *map) {
  if (map->map_flags & CL_MAP_WRITE_INVALIDATE_REGION)
    return CL_SUCCESS;

  pocl_simtsim_data_t *dd = (pocl_simtsim_data_t *)data;
  simtsim_buffer_data_t *b = (simtsim_buffer_data_t *)src_mem_id->mem_ptr;

  int err = simtsim_read_mem(dd->sim, map->host_ptr, b->addr + map->offset,
                             map->size);
  if (err != 0)
    POCL_ABORT("ERROR (pocl_simtsim_map_mem): Read from device failed\n");

  return CL_SUCCESS;
}

cl_int pocl_simtsim_unmap_mem(void *data, pocl_mem_identifier *dst_mem_id,
                              cl_mem dst_buf, mem_mapping_t *map) {
  if (map->map_flags == CL_MAP_READ)
    return CL_SUCCESS;

  pocl_simtsim_data_t *dd = (pocl_simtsim_data_t *)data;
  simtsim_buffer_data_t *b = (simtsim_buffer_data_t *)dst_mem_id->mem_ptr;

  int err = simtsim_write_mem(dd->sim, b->addr + map->offset, map->size,
                              map->host_ptr);
  if (err != 0)
    POCL_ABORT("ERROR (pocl_simtsim_unmap_mem): Write to device failed\n");

  return CL_SUCCESS;
}

static inline uint64_t align(uint64_t n, size_t size) {
  return (n + size - 1) & ~(size - 1);
}

void pocl_simtsim_run(void *data, _cl_command_node *cmd) {
  assert(data);
  pocl_simtsim_data_t *dd = (pocl_simtsim_data_t *)data;
  cl_uint device_i = cmd->program_device_i;
  cl_kernel kernel = cmd->command.run.kernel;
  cl_program program = kernel->program;
  pocl_kernel_metadata_t *meta = kernel->meta;
  simtsim_program_data_t *pdata =
      (simtsim_program_data_t *)program->data[device_i];
  SimtsimConfig cfg = simtsim_get_config(dd->sim);
  simtsim_kernel_data_t *kdata = (simtsim_kernel_data_t *)meta->data[device_i];
  struct pocl_context *context = &cmd->command.run.pc;
  int err = 0;

  uint32_t num_groups = 1;
  uint32_t group_size = 1;
  for (uint32_t i = 0; i < context->work_dim; ++i) {
    num_groups *= context->num_groups[i];
    group_size *= context->local_size[i];
  }
  if (num_groups == 0 || group_size == 0) {
    return;
  }

  const uint32_t ptr_size = 8;
  const uint32_t word_size = 8;

  size_t local_mem_size = 0;
  size_t kargs_buffer_size = word_size;

  for (int i = 0; i < meta->num_args; i++) {
    struct pocl_argument *arg = &(cmd->command.run.arguments[i]);
    if (ARG_IS_LOCAL(meta->arg_info[i])) {
      local_mem_size += arg->size;
      kargs_buffer_size = align(kargs_buffer_size, word_size) + word_size;
    } else if ((meta->arg_info[i].type == POCL_ARG_TYPE_POINTER) ||
               (meta->arg_info[i].type == POCL_ARG_TYPE_IMAGE) ||
               (meta->arg_info[i].type == POCL_ARG_TYPE_SAMPLER)) {
      kargs_buffer_size = align(kargs_buffer_size, ptr_size) + ptr_size;
    } else {
      kargs_buffer_size = align(kargs_buffer_size, arg->size) + arg->size;
    }
  }

  for (int i = 0; i < meta->num_locals; i++) {
    local_mem_size += meta->local_sizes[i];
    kargs_buffer_size = align(kargs_buffer_size, word_size) + word_size;
  }

  err = pocl_simtsim_check_kernel_valid(group_size);
  if (err != 0) {
    POCL_ABORT("ERROR (pocl_formosa_run): Check occupancy failed\n");
  }

  uint8_t *host_kargs_base_ptr = malloc(kargs_buffer_size);

  simtsim_buffer_data_t kargs_buffer;
  memset(&kargs_buffer, 0, sizeof(kargs_buffer));
  void *device_args_buffer_addr, *device_kernel_status_addr;
  err = fsa_malloc(&device_args_buffer_addr, kargs_buffer_size);
  if (err != 0) {
    POCL_ABORT("ERROR (pocl_formosa_run): Device Memory allocation failed\n");
  }

  kargs_buffer.addr = (uint64_t)device_args_buffer_addr;
  kargs_buffer.size = kargs_buffer_size;

  /* Write arguments */
  uint32_t host_args_offset = 0;
  uint64_t local_mem_offset = 0;

  if (local_mem_size > 0) {
    /* Total local mem size */
    memcpy(host_kargs_base_ptr + host_args_offset, &local_mem_size, word_size);
    host_args_offset += word_size;
  } else {
    memset(host_kargs_base_ptr + host_args_offset, 0, word_size);
    host_args_offset += word_size;
  }

  for (int i = 0; i < meta->num_args; ++i) {
    struct pocl_argument *al = &(cmd->command.run.arguments[i]);
    if (ARG_IS_LOCAL(meta->arg_info[i])) {
      host_args_offset = align(host_args_offset, word_size);
      /* local mem offset */
      memcpy(host_kargs_base_ptr + host_args_offset, &local_mem_offset,
             word_size);
      host_args_offset += word_size;
      local_mem_offset += al->size;
    } else if (meta->arg_info[i].type == POCL_ARG_TYPE_POINTER) {
      if (al->value == NULL) {
        /* NULL pointer value */
        host_args_offset = align(host_args_offset, ptr_size);
        memset(host_kargs_base_ptr + host_args_offset, 0, ptr_size);
        host_args_offset += ptr_size;
      } else {
        cl_mem m = (*(cl_mem *)(al->value));
        simtsim_buffer_data_t *buf_data =
            (simtsim_buffer_data_t *)m->device_ptrs[cmd->device->global_mem_id]
                .mem_ptr;
        uint64_t dev_mem_addr = buf_data->addr + al->offset;
        host_args_offset = align(host_args_offset, word_size);
        /* pointer value */
        memcpy(host_kargs_base_ptr + host_args_offset, &dev_mem_addr, ptr_size);
        host_args_offset += ptr_size;
      }
    } else if (meta->arg_info[i].type == POCL_ARG_TYPE_IMAGE) {
      POCL_ABORT("ERROR (pocl_formosa_run): Image argument not supported\n");
    } else if (meta->arg_info[i].type == POCL_ARG_TYPE_SAMPLER) {
      POCL_ABORT("ERROR (pocl_formosa_run): Sampler argument not supported\n");
    } else {
      /* scalar argument */
      host_args_offset = align(host_args_offset, al->size);
      memcpy(host_kargs_base_ptr + host_args_offset, al->value, al->size);
      host_args_offset += al->size;
    }
  }

  for (int i = 0; i < meta->num_locals; i++) {
    host_args_offset = align(host_args_offset, word_size);
    memcpy(host_kargs_base_ptr + host_args_offset, &local_mem_offset,
           word_size);
    host_args_offset += word_size;
    local_mem_offset += meta->local_sizes[i];
  }

  /* Upload kernel argument buffer */
  err = simtsim_write_mem(dd->sim, kargs_buffer.addr, kargs_buffer.size,
                          host_kargs_base_ptr);
  if (err != 0) {
    POCL_ABORT("ERROR (pocl_formosa_run): Kernel arg Copy to device failed\n");
  }
  free(host_kargs_base_ptr);

  /* Launch kernel */
  Dim3 local_size = {context->local_size[0], context->local_size[1],
                     context->local_size[2]};
  Dim3 global_size = {context->num_groups[0] * local_size.x,
                      context->num_groups[1] * local_size.y,
                      context->num_groups[2] * local_size.z};
  size_t num_blocks =
      context->num_groups[0] * context->num_groups[1] * context->num_groups[2];

  const uint64_t stack_size = 0x200;
  void *stack_addr;
  size_t stack_bytes = num_blocks * group_size * stack_size;
  err = fsa_malloc(&stack_addr, stack_bytes);
  if (err != 0) {
    POCL_ABORT("ERROR (pocl_formosa_run): Stack allocation failed\n");
  }
  simtsim_set_pointer(dd->sim, SIMTSIM_PTR_STACK_BASE, (uint64_t)stack_addr);
  simtsim_set_pointer(dd->sim, SIMTSIM_PTR_STACK_SIZE, stack_size);

  /* Use global mem as local mem. Allocate private local mem for each block */
  void *local_mem_addr;
  if (local_mem_size > 0) {
    err = fsa_malloc(&local_mem_addr, num_blocks * local_mem_size);
    if (err != 0) {
      POCL_ABORT("ERROR (pocl_formosa_run): Local memory allocation failed\n");
    }
    simtsim_set_pointer(dd->sim, SIMTSIM_PTR_LOCAL_MEM_BASE,
                        (uint64_t)local_mem_addr);
    simtsim_set_pointer(dd->sim, SIMTSIM_PTR_LOCAL_MEM_SIZE, local_mem_size);
  }

  /* Allocate one time use info addr, `simtsim::launch_kernel` will populate the
   * structure, we dont need to do it here */
  void *device_info_addr;
  size_t info_size = -1;
  if (cfg.cuda) {
    /* We cannot know how thread blocks (workgroup) are executed on the GPU.
     * Therefore we allocate the maximum size to avoid WGInfo race condition */
    info_size = num_blocks * sizeof(WGInfo);
  } else if (cfg.mt2) {
    info_size = cfg.num_threads * sizeof(WGInfo);
  } else {
    info_size = sizeof(WGInfo);
  }
  assert(info_size >= sizeof(WGInfo));
  err = fsa_malloc(&device_info_addr, info_size);
  if (err != 0) {
    POCL_ABORT(
        "ERROR (pocl_formosa_run): Device info buffer allocation failed\n");
  }
  simtsim_set_pointer(dd->sim, SIMTSIM_PTR_INFO, (uint64_t)device_info_addr);
  simtsim_set_pointer(dd->sim, SIMTSIM_PTR_ARG, kargs_buffer.addr);

  /* The compiler generates a <kernel>_trampoline wrapper */
  char trampoline_name[256];
  snprintf(trampoline_name, sizeof(trampoline_name), "%s_trampolined",
           kernel->name);

  if (cfg.cuda) {
    simtsim_launch_kernel_cu(dd->sim, pdata->sim_program, trampoline_name,
                             global_size, local_size);
  } else if (cfg.mt2) {
    simtsim_launch_kernel_mt(dd->sim, pdata->sim_program, trampoline_name,
                             global_size, local_size);
  } else {
    simtsim_launch_kernel(dd->sim, pdata->sim_program, trampoline_name,
                          global_size, local_size);
  }

  /* release kernel arg buf */
  err = fsa_free((void *)kargs_buffer.addr);
  if (err != 0) {
    POCL_ABORT("ERROR (pocl_formosa_run): Kernel arg free failed\n");
  }
  err = fsa_free(device_info_addr);
  if (err != 0) {
    POCL_ABORT("ERROR (pocl_formosa_run): Info buffer free failed\n");
  }
  if (local_mem_size > 0) {
    err = fsa_free(local_mem_addr);
    if (err != 0) {
      POCL_ABORT("ERROR (pocl_formosa_run): Info buffer free failed\n");
    }
  }
  err = fsa_free(stack_addr);
  if (err != 0) {
    POCL_ABORT("ERROR (pocl_formosa_run): Stack free failed\n");
  }
}
