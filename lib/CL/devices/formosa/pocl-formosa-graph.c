#include "pocl_cl.h"
#include "pocl_util.h"
#include "formosa-hal/formosa-hal.h"
#include "formosa-hal/formosa-graph.h"
#include "formosa-util.h"
#include "formosa-llvm-util.h"
#include "pocl-formosa.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

struct pocl_formosa_work_graph_data {
  uint64_t dev_graph_status;
  uint64_t dev_graph_desc;
  uint64_t dev_node_descs;
  uint64_t dev_runtime_pool;
};

struct pocl_formosa_work_graph_node_data {
  uint64_t dev_static_kargs;
};

cl_int pocl_formosa_create_work_graph(
    cl_work_graph_formosa graph,
    const cl_work_graph_properties_formosa *properties) {
  struct pocl_formosa_work_graph_data *bg =
      (struct pocl_formosa_work_graph_data *)calloc(
          1, sizeof(struct pocl_formosa_work_graph_data));
  if (bg == NULL) return CL_OUT_OF_HOST_MEMORY;

  graph->backend_data = bg;
  return CL_SUCCESS;
}

cl_int pocl_formosa_create_work_graph_node(
    cl_work_graph_node_formosa node, cl_kernel kernel, cl_uint node_id,
    cl_uint work_dim, const size_t *global_offset, const size_t *global_size,
    const size_t *local_size,
    const cl_work_graph_node_properties_formosa *properties) {
  struct pocl_formosa_work_graph_node_data *bn =
      (struct pocl_formosa_work_graph_node_data *)calloc(
          1, sizeof(struct pocl_formosa_work_graph_node_data));
  if (bn == NULL) return CL_OUT_OF_HOST_MEMORY;

  node->backend_data = bn;
  return CL_SUCCESS;
}

cl_int pocl_formosa_create_work_graph_edge(
    cl_work_graph_edge_formosa edge, cl_work_graph_node_formosa src,
    cl_work_graph_node_formosa dst, cl_uint edge_id,
    const cl_work_graph_edge_properties_formosa *properties) {
  /* Not implemented in Phase 1 */
  return CL_SUCCESS;
}

static cl_int pocl_formosa_prepare_mem_arg(
    cl_device_id device, cl_mem mem, formosa_buffer_data_t **buf_data_out) {
  if (mem == NULL) return CL_INVALID_MEM_OBJECT;
  if (mem->device_ptrs == NULL) return CL_INVALID_MEM_OBJECT;

  /* If the memory object hasn't been allocated on the device yet (e.g., lazy 
     allocation for CL_MEM_WRITE_ONLY buffers), allocate it now. */
  if (mem->device_ptrs[device->global_mem_id].mem_ptr == NULL) {
    if (device->ops->alloc_mem_obj != NULL) {
      cl_int err = device->ops->alloc_mem_obj(device, mem, NULL);
      if (err != CL_SUCCESS) {
        return CL_MEM_OBJECT_ALLOCATION_FAILURE;
      }
    } else {
      return CL_INVALID_OPERATION;
    }
  }

  formosa_buffer_data_t *buf_data =
      (formosa_buffer_data_t *)mem->device_ptrs[device->global_mem_id].mem_ptr;

  if (buf_data == NULL) return CL_INVALID_MEM_OBJECT;
  
  if (buf_data_out) *buf_data_out = buf_data;
  return CL_SUCCESS;
}

/* Phase 1: pack the single output buffer argument.
 * TODO: generalize this helper to iterate over kernel metadata and pack all
 * pointer/scalar/local arguments exactly like pocl_formosa_run().
 */
static cl_int pocl_formosa_pack_kernel_args(cl_device_id device,
                                            cl_kernel kernel,
                                            uint64_t *dev_kernarg_addr,
                                            uint32_t *kernarg_size,
                                            uint64_t *local_mem_size) {
  if (kernel == NULL) return CL_INVALID_KERNEL;
  if (kernel->dyn_arguments == NULL) return CL_INVALID_KERNEL_ARGS;

  size_t kargs_host_size = 64; /* Enough for Phase 1 */
  uint8_t host_kargs[64] = {0};
  uint32_t host_args_offset = 24; // PoclPrintfLaunchMeta
  uint64_t l_mem_size = 0;

  /* PoclPrintfLaunchMeta and Local Mem Size placeholder */
  memcpy(host_kargs + host_args_offset, &l_mem_size, 8);
  host_args_offset += 8;

  /* Fill arg0 (output buffer pointer) */
  struct pocl_argument *arg0 = &kernel->dyn_arguments[0];
  if (arg0->value == NULL) return CL_INVALID_KERNEL_ARGS;

  cl_mem out_mem = *(cl_mem *)arg0->value;
  formosa_buffer_data_t *buf_data = NULL;
  cl_int err = pocl_formosa_prepare_mem_arg(device, out_mem, &buf_data);
  if (err != CL_SUCCESS) return err;

  uint64_t out_dev_addr = buf_data->buf_address + arg0->offset;
  memcpy(host_kargs + host_args_offset, &out_dev_addr, 8);

  void *addr;
  int fsa_err = fsa_malloc(&addr, kargs_host_size);
  if (fsa_err) return CL_OUT_OF_RESOURCES;

  fsa_copy_to_dev((uintptr_t)addr, host_kargs, kargs_host_size);

  *dev_kernarg_addr = (uint64_t)addr;
  *kernarg_size = (uint32_t)kargs_host_size;
  *local_mem_size = l_mem_size;

  return CL_SUCCESS;
}

void pocl_formosa_run_work_graph(void *data, _cl_command_node *cmd) {
  cl_work_graph_formosa graph =
      (cl_work_graph_formosa)cmd->command.work_graph_launch_formosa.graph;
  cl_uint num_root_inputs =
      cmd->command.work_graph_launch_formosa.num_root_inputs;
  cl_work_graph_root_input_formosa *root_inputs =
      (cl_work_graph_root_input_formosa *)
          cmd->command.work_graph_launch_formosa.root_inputs;

  struct pocl_formosa_work_graph_data *bg =
      (struct pocl_formosa_work_graph_data *)graph->backend_data;
  cl_device_id device = cmd->device;
  pocl_formosa_data_t *dd = (pocl_formosa_data_t *)data;

  /* Phase 1 Validation */
  cl_uint node_count = 0;
  struct _cl_work_graph_node_formosa *curr_node = graph->nodes;
  while (curr_node) {
    node_count++;
    curr_node = curr_node->next;
  }
  if (node_count != 1) {
    POCL_MSG_ERR("formosa: Phase 1 graph must have exactly 1 node\n");
    return;
  }
  if (graph->edges != NULL) {
    POCL_MSG_ERR("formosa: Phase 1 graph must not have edges\n");
    return;
  }
  if (num_root_inputs != 1) {
    POCL_MSG_ERR("formosa: Phase 1 graph must have exactly 1 root input\n");
    return;
  }

  struct _cl_work_graph_node_formosa *found_node = graph->nodes;
  struct pocl_formosa_work_graph_node_data *bn =
      (struct pocl_formosa_work_graph_node_data *)found_node->backend_data;

  /* Resolve kernel symbols */
  char sz_program_fsabin[POCL_MAX_PATHNAME_LENGTH];
  pocl_fsa_get_elf_name(found_node->kernel->program, 0, sz_program_fsabin);

  uint64_t dev_kernel_addr = 0;
  pocl_fsa_upload_kernel(sz_program_fsabin, dd, &dev_kernel_addr);

  uint64_t entry_pc = pocl_fsa_get_symbol_pc(sz_program_fsabin, "_start") + dev_kernel_addr;
  char trampoline_name[256];
  snprintf(trampoline_name, sizeof(trampoline_name), "%s_trampoline", found_node->kernel->name);
  uint64_t trampoline_pc = pocl_fsa_get_symbol_pc(sz_program_fsabin, trampoline_name) + dev_kernel_addr;

  /* Allocate/Update Device Descriptors */
  if (bg->dev_graph_status == 0) {
    fsa_malloc((void **)&bg->dev_graph_status, sizeof(GraphStatus));
    fsa_malloc((void **)&bg->dev_graph_desc, sizeof(GraphDescriptor));
    fsa_malloc((void **)&bg->dev_node_descs, sizeof(NodeDescriptor));
    fsa_malloc((void **)&bg->dev_runtime_pool, sizeof(GraphRuntimePool));
  }

  /* Fill Node Descriptor */
  NodeDescriptor nd = {0};
  nd.node_id = found_node->node_id;
  nd.launch_mode = found_node->properties.launch_mode;
  nd.input_record_size = found_node->properties.input_record_size;
  nd.kernel_object = entry_pc;
  nd.kernel_trampoline = trampoline_pc;
  nd.work_dim = found_node->work_dim;
  for (int i = 0; i < 3; i++) {
    nd.global_work_offset[i] = found_node->global_work_offset[i];
    nd.global_work_size[i] = found_node->global_work_size[i];
    nd.local_work_size[i] = found_node->local_work_size[i];
  }

  /* Prepare static kernargs for the node kernel */
  if (bn->dev_static_kargs == 0) {
    uint64_t dev_kargs_addr = 0;
    uint32_t kargs_size = 0;
    uint64_t local_mem_size = 0;

    cl_int err = pocl_formosa_pack_kernel_args(device, found_node->kernel,
                                               &dev_kargs_addr, &kargs_size,
                                               &local_mem_size);
    if (err != CL_SUCCESS) {
        POCL_MSG_ERR("formosa: failed to pack kernel args for graph node\n");
        return;
    }

    bn->dev_static_kargs = dev_kargs_addr;
  }

  nd.static_kernarg_addr = bn->dev_static_kargs;
  nd.static_kernarg_size = 64; // Matches kargs_host_size in pack_helper

  fsa_copy_to_dev(bg->dev_node_descs, &nd, sizeof(NodeDescriptor));

  /* Fill Graph Descriptor */
  GraphDescriptor gd = {0};
  gd.version = 1;
  gd.node_count = 1;
  gd.edge_count = 0;
  gd.node_desc_addr = bg->dev_node_descs;
  fsa_copy_to_dev(bg->dev_graph_desc, &gd, sizeof(GraphDescriptor));

  /* Fill Root Input Descriptor */
  RootInputDescriptor rid = {0};
  rid.target_node_id = root_inputs[0].target_node_id;
  rid.record_count = root_inputs[0].record_count;
  
  formosa_buffer_data_t *root_buf_data = NULL;
  cl_int err = pocl_formosa_prepare_mem_arg(device, root_inputs[0].records,
                                            &root_buf_data);
  if (err != CL_SUCCESS) {
      POCL_MSG_ERR("formosa: failed to prepare root input memory\n");
      return;
  }

  rid.records_addr = root_buf_data->buf_address;
  rid.records_offset = root_inputs[0].records_offset;
  rid.record_stride = root_inputs[0].record_stride;

  uint64_t dev_root_desc = 0;
  fsa_malloc((void **)&dev_root_desc, sizeof(RootInputDescriptor));
  fsa_copy_to_dev(dev_root_desc, &rid, sizeof(RootInputDescriptor));

  /* Fill Runtime Pool */
  GraphRuntimePool pool = {0};
  fsa_malloc((void **)&pool.ctx_pool_base, 256); /* Enough for 1 ctx */
  fsa_malloc((void **)&pool.kernarg_pool_base, 256); /* Enough for 1 kargs */
  fsa_copy_to_dev(bg->dev_runtime_pool, &pool, sizeof(GraphRuntimePool));

  /* Launch */
  uintptr_t completion_signal = 0;

  int launch_rc = fsa_cmd_launch_graph(0, rid.record_count,
                                      bg->dev_graph_desc,
                                      bg->dev_runtime_pool,
                                      bg->dev_graph_status,
                                      dev_root_desc,
                                      rid.records_addr + rid.records_offset,
                                      &completion_signal);

  /* Wait for completion (Temporary synchronous wait for Phase 1) */
  if (launch_rc == 0) {
    pocl_fsa_wait_ack(dd, completion_signal, 0);
  } else {
    POCL_MSG_ERR("formosa: graph launch failed\n");
  }

  /* Clean up temporary allocations */
  fsa_free((void *)dev_root_desc);
  fsa_free((void *)pool.ctx_pool_base);
  fsa_free((void *)pool.kernarg_pool_base);
}

cl_int pocl_formosa_get_work_graph_info(cl_work_graph_formosa graph,
                                        cl_uint param, size_t size, void *value,
                                        size_t *size_ret) {
  if (param == CL_GRAPH_INFO_STATUS_FORMOSA) {
    if (value) {
      if (size < sizeof(cl_uint)) return CL_INVALID_VALUE;
      *(cl_uint *)value = CL_GRAPH_STATUS_IDLE_FORMOSA;
    }
    if (size_ret) *size_ret = sizeof(cl_uint);
    return CL_SUCCESS;
  }
  return CL_INVALID_VALUE;
}

cl_int pocl_formosa_free_work_graph(cl_work_graph_formosa graph) {
  struct pocl_formosa_work_graph_data *bg =
      (struct pocl_formosa_work_graph_data *)graph->backend_data;
  if (bg) {
    if (bg->dev_graph_status) fsa_free((void *)bg->dev_graph_status);
    if (bg->dev_graph_desc) fsa_free((void *)bg->dev_graph_desc);
    if (bg->dev_node_descs) fsa_free((void *)bg->dev_node_descs);
    if (bg->dev_runtime_pool) fsa_free((void *)bg->dev_runtime_pool);
    free(bg);
  }

  /* Core will free node wrappers, but we should free backend node data */
  struct _cl_work_graph_node_formosa *curr_node = graph->nodes;
  while (curr_node) {
    struct pocl_formosa_work_graph_node_data *bn =
        (struct pocl_formosa_work_graph_node_data *)curr_node->backend_data;
    if (bn) {
      if (bn->dev_static_kargs) fsa_free((void *)bn->dev_static_kargs);
      free(bn);
    }
    curr_node = curr_node->next;
  }

  return CL_SUCCESS;
}
