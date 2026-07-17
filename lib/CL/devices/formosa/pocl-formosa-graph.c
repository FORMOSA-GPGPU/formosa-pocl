#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "formosa-hal/formosa-graph.h"
#include "formosa-hal/formosa-hal.h"
#include "formosa-llvm-util.h"
#include "formosa-util.h"
#include "pocl-formosa.h"
#include "pocl_cl.h"
#include "pocl_util.h"

/* Extra ctx/kernarg slots let firmware keep multiple dispatches active for the
   same graph, including same-node dispatch overlap. The hard firmware cap is
   still enforced by CP_GRAPH_MAX_DISPATCH_SLOTS. */
#define FORMOSA_WG_EXTRA_DISPATCH_SLOTS 4
#define FORMOSA_WG_MAX_DISPATCH_SLOTS 64
#define FORMOSA_WG_DEFAULT_QUEUE_CAPACITY 16

static inline uint64_t pocl_formosa_align(uint64_t n, size_t size) {
  return (n + size - 1) & ~(size - 1);
}

static inline uint32_t pocl_formosa_default_queue_capacity(
    cl_uint root_record_count) {
  return root_record_count != 0 ? root_record_count
                                : FORMOSA_WG_DEFAULT_QUEUE_CAPACITY;
}

static int pocl_formosa_mul_u32_overflow(uint32_t a, uint32_t b,
                                         uint32_t *out) {
  if (a != 0 && b > UINT32_MAX / a) return 1;
  *out = a * b;
  return 0;
}

static cl_int pocl_formosa_add_graph_cache_range(
    GraphCacheRange **ranges, uint32_t *count, uint32_t *capacity,
    uint64_t addr, uint64_t size, uint32_t flags, uint32_t node_id) {
  if (addr == 0 || size == 0 || addr > UINT64_MAX - size)
    return CL_INVALID_VALUE;

  for (uint32_t i = 0; i < *count; i++) {
    if ((*ranges)[i].addr == addr && (*ranges)[i].size == size &&
        (*ranges)[i].flags == flags && (*ranges)[i].node_id == node_id)
      return CL_SUCCESS;
  }

  if (*count == *capacity) {
    if (*capacity > UINT32_MAX / 2) return CL_OUT_OF_HOST_MEMORY;
    uint32_t new_capacity = *capacity == 0 ? 8 : *capacity * 2;
    GraphCacheRange *new_ranges = (GraphCacheRange *)realloc(
        *ranges, (size_t)new_capacity * sizeof(**ranges));
    if (new_ranges == NULL) return CL_OUT_OF_HOST_MEMORY;
    *ranges = new_ranges;
    *capacity = new_capacity;
  }

  (*ranges)[*count].addr = addr;
  (*ranges)[*count].size = size;
  (*ranges)[*count].flags = flags;
  (*ranges)[*count].node_id = node_id;
  (*count)++;
  return CL_SUCCESS;
}

static cl_int pocl_formosa_validate_node_properties(
    const struct _cl_work_graph_node_formosa *node) {
  uint32_t launch_mode = node->properties.launch_mode;
  if (launch_mode != CL_NODE_LAUNCH_THREAD_FORMOSA &&
      launch_mode != CL_NODE_LAUNCH_COALESCING_FORMOSA &&
      launch_mode != CL_NODE_LAUNCH_BROADCASTING_FORMOSA) {
    POCL_MSG_ERR("formosa: unsupported WorkGraph node launch mode %u\n",
                 launch_mode);
    return CL_INVALID_VALUE;
  }

  if (launch_mode != CL_NODE_LAUNCH_BROADCASTING_FORMOSA &&
      node->properties.work_items_per_input_record > 1) {
    POCL_MSG_ERR(
        "formosa: work_items_per_input_record requires BROADCASTING mode\n");
    return CL_INVALID_VALUE;
  }

  if (launch_mode == CL_NODE_LAUNCH_COALESCING_FORMOSA &&
      node->properties.max_input_records_per_dispatch == 0) {
    POCL_MSG_ERR(
        "formosa: COALESCING nodes require max_input_records_per_dispatch\n");
    return CL_INVALID_VALUE;
  }

  if (node->properties.input_record_size == 0) {
    POCL_MSG_ERR("formosa: graph node %u has zero input record size\n",
                 node->node_id);
    return CL_INVALID_VALUE;
  }

  return CL_SUCCESS;
}

struct pocl_formosa_work_graph_data {
  uint64_t dev_graph_status;
  uint64_t dev_graph_desc;
  uint64_t dev_node_descs;
  uint64_t dev_edge_descs;
  uint64_t dev_node_queues;
  uint64_t dev_node_states;
  uint64_t dev_runtime_pool;

  uint64_t *dev_node_queue_records;
  uint64_t *dev_node_queue_ready_sequences;
  size_t *node_queue_record_sizes;
  size_t *node_queue_ready_sequence_sizes;
  cl_uint node_desc_capacity;
  cl_uint edge_desc_capacity;
  cl_uint node_queue_capacity;
  cl_uint node_state_capacity;
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
  cl_int err = pocl_formosa_validate_node_properties(node);
  if (err != CL_SUCCESS) return err;

  return CL_SUCCESS;
}

cl_int pocl_formosa_create_work_graph_edge(
    cl_work_graph_edge_formosa edge, cl_work_graph_node_formosa src,
    cl_work_graph_node_formosa dst, cl_uint edge_id,
    const cl_work_graph_edge_properties_formosa *properties) {
  /* No Formosa-specific edge backend data is needed yet. */
  return CL_SUCCESS;
}

static void pocl_formosa_free_node_queue_storage(
    struct pocl_formosa_work_graph_data *bg) {
  if (bg->dev_node_queue_records) {
    for (cl_uint i = 0; i < bg->node_queue_capacity; i++) {
      if (bg->dev_node_queue_records[i])
        fsa_free((void *)bg->dev_node_queue_records[i]);
    }
    free(bg->dev_node_queue_records);
    bg->dev_node_queue_records = NULL;
  }

  if (bg->dev_node_queue_ready_sequences) {
    for (cl_uint i = 0; i < bg->node_queue_capacity; i++) {
      if (bg->dev_node_queue_ready_sequences[i])
        fsa_free((void *)bg->dev_node_queue_ready_sequences[i]);
    }
    free(bg->dev_node_queue_ready_sequences);
    bg->dev_node_queue_ready_sequences = NULL;
  }

  free(bg->node_queue_record_sizes);
  bg->node_queue_record_sizes = NULL;
  free(bg->node_queue_ready_sequence_sizes);
  bg->node_queue_ready_sequence_sizes = NULL;
  bg->node_queue_capacity = 0;
}

static int pocl_formosa_find_node_index_by_ptr(
    struct _cl_work_graph_node_formosa **nodes, cl_uint node_count,
    struct _cl_work_graph_node_formosa *node) {
  for (cl_uint i = 0; i < node_count; i++) {
    if (nodes[i] == node) return (int)i;
  }
  return -1;
}

static int pocl_formosa_graph_has_cycle(
    struct _cl_work_graph_node_formosa **nodes, cl_uint node_count,
    struct _cl_work_graph_edge_formosa **edges, cl_uint edge_count) {
  cl_uint *indegree = (cl_uint *)calloc(node_count, sizeof(cl_uint));
  cl_uint *queue = (cl_uint *)calloc(node_count, sizeof(cl_uint));
  if (indegree == NULL || queue == NULL) {
    free(indegree);
    free(queue);
    return -1;
  }

  for (cl_uint i = 0; i < edge_count; i++) {
    int dst_idx = pocl_formosa_find_node_index_by_ptr(nodes, node_count,
                                                      edges[i]->dst_node);
    if (dst_idx >= 0) indegree[dst_idx]++;
  }

  cl_uint head = 0;
  cl_uint tail = 0;
  for (cl_uint i = 0; i < node_count; i++) {
    if (indegree[i] == 0) queue[tail++] = i;
  }

  cl_uint visited = 0;
  while (head < tail) {
    cl_uint src_idx = queue[head++];
    visited++;
    for (cl_uint i = 0; i < edge_count; i++) {
      int edge_src_idx = pocl_formosa_find_node_index_by_ptr(
          nodes, node_count, edges[i]->src_node);
      if (edge_src_idx != (int)src_idx) continue;

      int dst_idx = pocl_formosa_find_node_index_by_ptr(nodes, node_count,
                                                        edges[i]->dst_node);
      if (dst_idx >= 0 && --indegree[dst_idx] == 0)
        queue[tail++] = (cl_uint)dst_idx;
    }
  }

  free(indegree);
  free(queue);
  return visited != node_count;
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

static cl_int pocl_formosa_pack_kernel_args(
    cl_device_id device, cl_kernel kernel, uint32_t node_id,
    uint32_t node_flags, uint64_t *dev_kernarg_addr, uint32_t *kernarg_size,
    uint64_t *local_mem_size, GraphCacheRange **cache_ranges,
    uint32_t *cache_range_count, uint32_t *cache_range_capacity) {
  if (kernel == NULL) return CL_INVALID_KERNEL;
  if (kernel->dyn_arguments == NULL) return CL_INVALID_KERNEL_ARGS;

  pocl_kernel_metadata_t *meta = kernel->meta;
  if (meta == NULL) return CL_INVALID_KERNEL;

  const uint32_t ptr_size = 8;
  const uint32_t word_size = 8;
  /* Graph printf is unsupported; keep this zeroed prefix for trampoline ABI. */
  const uint32_t printf_meta_size = sizeof(formosa_printf_launch_meta_t);
  uint64_t l_mem_size = 0;

  size_t kargs_host_size = printf_meta_size + word_size;
  for (int i = 0; i < meta->num_args; i++) {
    struct pocl_argument *arg = &kernel->dyn_arguments[i];
    if (ARG_IS_LOCAL(meta->arg_info[i])) {
      l_mem_size += arg->size;
      kargs_host_size =
          pocl_formosa_align(kargs_host_size, word_size) + word_size;
    } else if (meta->arg_info[i].type == POCL_ARG_TYPE_POINTER ||
               meta->arg_info[i].type == POCL_ARG_TYPE_IMAGE ||
               meta->arg_info[i].type == POCL_ARG_TYPE_SAMPLER) {
      kargs_host_size =
          pocl_formosa_align(kargs_host_size, ptr_size) + ptr_size;
    } else {
      kargs_host_size =
          pocl_formosa_align(kargs_host_size, arg->size) + arg->size;
    }
  }

  for (int i = 0; i < meta->num_locals; i++) {
    l_mem_size += meta->local_sizes[i];
    kargs_host_size =
        pocl_formosa_align(kargs_host_size, word_size) + word_size;
  }

  uint8_t *host_kargs = (uint8_t *)calloc(1, kargs_host_size);
  if (host_kargs == NULL) return CL_OUT_OF_HOST_MEMORY;

  uint32_t host_args_offset = printf_meta_size;
  uint64_t local_mem_offset = 0;

  memcpy(host_kargs + host_args_offset, &l_mem_size, word_size);
  host_args_offset += word_size;

  for (int i = 0; i < meta->num_args; i++) {
    struct pocl_argument *arg = &kernel->dyn_arguments[i];

    if (ARG_IS_LOCAL(meta->arg_info[i])) {
      host_args_offset = pocl_formosa_align(host_args_offset, word_size);
      memcpy(host_kargs + host_args_offset, &local_mem_offset, word_size);
      host_args_offset += word_size;
      local_mem_offset += arg->size;
    } else if (meta->arg_info[i].type == POCL_ARG_TYPE_POINTER) {
      host_args_offset = pocl_formosa_align(host_args_offset, ptr_size);
      uint64_t dev_addr = 0;

      if (arg->value != NULL) {
        cl_mem mem = *(cl_mem *)arg->value;
        formosa_buffer_data_t *buf_data = NULL;
        cl_int err = pocl_formosa_prepare_mem_arg(device, mem, &buf_data);
        if (err != CL_SUCCESS) {
          free(host_kargs);
          return err;
        }
        dev_addr = buf_data->buf_address + arg->offset;
        uint32_t range_flags = 0;
        uint32_t range_node_id = FORMOSA_GRAPH_CACHE_RANGE_ALL_NODES;
        if ((meta->arg_info[i].type_qualifier & CL_KERNEL_ARG_TYPE_CONST) ==
            0) {
          range_flags = FORMOSA_GRAPH_CACHE_RANGE_COMPLETION;
        } else if ((node_flags & CL_NODE_GLOBAL_CACHE_BEFORE_FORMOSA) != 0) {
          range_flags = FORMOSA_GRAPH_CACHE_RANGE_NODE_BEFORE;
          range_node_id = node_id;
        }
        if (range_flags != 0) {
          err = pocl_formosa_add_graph_cache_range(
              cache_ranges, cache_range_count, cache_range_capacity,
              buf_data->buf_address, buf_data->buf_size, range_flags,
              range_node_id);
        }
        if (err != CL_SUCCESS) {
          free(host_kargs);
          return err;
        }
      }

      memcpy(host_kargs + host_args_offset, &dev_addr, ptr_size);
      host_args_offset += ptr_size;
    } else if (meta->arg_info[i].type == POCL_ARG_TYPE_IMAGE ||
               meta->arg_info[i].type == POCL_ARG_TYPE_SAMPLER) {
      free(host_kargs);
      return CL_INVALID_ARG_VALUE;
    } else {
      if (arg->value == NULL) {
        free(host_kargs);
        return CL_INVALID_KERNEL_ARGS;
      }

      host_args_offset = pocl_formosa_align(host_args_offset, arg->size);
      memcpy(host_kargs + host_args_offset, arg->value, arg->size);
      host_args_offset += arg->size;
    }
  }

  for (int i = 0; i < meta->num_locals; i++) {
    host_args_offset = pocl_formosa_align(host_args_offset, word_size);
    memcpy(host_kargs + host_args_offset, &local_mem_offset, word_size);
    host_args_offset += word_size;
    local_mem_offset += meta->local_sizes[i];
  }

  void *addr;
  int fsa_err = fsa_malloc(&addr, kargs_host_size);
  if (fsa_err) {
    free(host_kargs);
    return CL_OUT_OF_RESOURCES;
  }

  fsa_err = fsa_copy_to_dev((uintptr_t)addr, host_kargs, kargs_host_size);
  free(host_kargs);
  if (fsa_err) {
    fsa_free(addr);
    return CL_OUT_OF_RESOURCES;
  }

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

  struct _cl_work_graph_node_formosa **listed_nodes = NULL;
  struct _cl_work_graph_node_formosa **ordered_nodes = NULL;
  struct _cl_work_graph_edge_formosa **ordered_edges = NULL;
  uint32_t *node_queue_caps = NULL;
  NodeDescriptor *node_descs = NULL;
  EdgeDescriptor *edge_descs = NULL;
  NodeInputQueue *node_queues = NULL;
  NodeRuntimeState *node_states = NULL;
  uint8_t *edge_capacity_resolved = NULL;
  uint32_t *node_unresolved_in_edges = NULL;
  uint64_t *launch_static_kargs = NULL;
  uint32_t *launch_static_karg_sizes = NULL;
  uint64_t *launch_local_mem_sizes = NULL;
  GraphCacheRange *cache_ranges = NULL;
  uint32_t cache_range_count = 0;
  uint32_t cache_range_capacity = 0;
  uint64_t dev_cache_ranges = 0;
  uint64_t dev_root_desc = 0;
  GraphRuntimePool pool = {0};
  cl_int err = CL_SUCCESS;

  cl_uint node_count = 0;
  struct _cl_work_graph_node_formosa *curr_node = graph->nodes;
  while (curr_node) {
    node_count++;
    curr_node = curr_node->next;
  }

  cl_uint edge_count = 0;
  struct _cl_work_graph_edge_formosa *curr_edge = graph->edges;
  while (curr_edge) {
    edge_count++;
    curr_edge = curr_edge->next;
  }

  if (node_count == 0) {
    POCL_MSG_ERR("formosa: graph launch requires at least one node\n");
    return;
  }
  if (node_count > FORMOSA_WG_MAX_DISPATCH_SLOTS) {
    POCL_MSG_ERR("formosa: graph has too many nodes (%u > %u)\n", node_count,
                 FORMOSA_WG_MAX_DISPATCH_SLOTS);
    return;
  }

  if (num_root_inputs != 1) {
    POCL_MSG_ERR("formosa: graph launch must have exactly 1 root input\n");
    return;
  }

  struct _cl_work_graph_node_formosa *root_node = NULL;
  curr_node = graph->nodes;
  while (curr_node) {
    if (curr_node->node_id == root_inputs[0].target_node_id)
      root_node = curr_node;
    curr_node = curr_node->next;
  }

  if (root_node == NULL) {
    POCL_MSG_ERR("formosa: root target node not found\n");
    return;
  }
  if ((root_node->properties.flags & CL_NODE_ROOT_CAPABLE_FORMOSA) == 0) {
    POCL_MSG_ERR("formosa: root target node %u is not root-capable\n",
                 root_node->node_id);
    return;
  }

  listed_nodes = (struct _cl_work_graph_node_formosa **)calloc(
      node_count, sizeof(*listed_nodes));
  ordered_nodes = (struct _cl_work_graph_node_formosa **)calloc(
      node_count, sizeof(*ordered_nodes));
  ordered_edges = edge_count ? (struct _cl_work_graph_edge_formosa **)calloc(
                                   edge_count, sizeof(*ordered_edges))
                             : NULL;
  node_queue_caps = (uint32_t *)calloc(node_count, sizeof(*node_queue_caps));
  node_descs = (NodeDescriptor *)calloc(node_count, sizeof(*node_descs));
  edge_descs = edge_count
                   ? (EdgeDescriptor *)calloc(edge_count, sizeof(*edge_descs))
                   : NULL;
  node_queues = (NodeInputQueue *)calloc(node_count, sizeof(*node_queues));
  node_states = (NodeRuntimeState *)calloc(node_count, sizeof(*node_states));
  edge_capacity_resolved =
      edge_count ? (uint8_t *)calloc(edge_count, sizeof(uint8_t)) : NULL;
  node_unresolved_in_edges =
      (uint32_t *)calloc(node_count, sizeof(*node_unresolved_in_edges));
  launch_static_kargs =
      (uint64_t *)calloc(node_count, sizeof(*launch_static_kargs));
  launch_static_karg_sizes =
      (uint32_t *)calloc(node_count, sizeof(*launch_static_karg_sizes));
  launch_local_mem_sizes =
      (uint64_t *)calloc(node_count, sizeof(*launch_local_mem_sizes));

  if (!listed_nodes || !ordered_nodes || (edge_count && !ordered_edges) ||
      !node_queue_caps || !node_descs || (edge_count && !edge_descs) ||
      !node_queues || !node_states || (edge_count && !edge_capacity_resolved) ||
      !node_unresolved_in_edges || !launch_static_kargs ||
      !launch_static_karg_sizes || !launch_local_mem_sizes) {
    POCL_MSG_ERR("formosa: out of host memory while lowering graph\n");
    goto CLEANUP;
  }

  cl_uint node_idx = 0;
  curr_node = graph->nodes;
  while (curr_node) {
    listed_nodes[node_idx++] = curr_node;
    curr_node = curr_node->next;
  }

  /* The public graph list is prepended, so walking it directly reverses API
     creation order. Preserve creation order for deterministic queue scans after
     the root. */
  ordered_nodes[0] = root_node;
  node_idx = 1;
  for (cl_uint rev_i = node_count; rev_i > 0; rev_i--) {
    curr_node = listed_nodes[rev_i - 1];
    if (curr_node != root_node) ordered_nodes[node_idx++] = curr_node;
  }

  for (cl_uint i = 0; i < node_count; i++) {
    err = pocl_formosa_validate_node_properties(ordered_nodes[i]);
    if (err != CL_SUCCESS) {
      goto CLEANUP;
    }
    for (cl_uint j = i + 1; j < node_count; j++) {
      if (ordered_nodes[i]->node_id == ordered_nodes[j]->node_id) {
        POCL_MSG_ERR("formosa: duplicate graph node id %u\n",
                     ordered_nodes[i]->node_id);
        goto CLEANUP;
      }
    }
  }

  cl_uint edge_idx = 0;
  curr_edge = graph->edges;
  while (curr_edge) {
    ordered_edges[edge_idx++] = curr_edge;
    curr_edge = curr_edge->next;
  }

  for (cl_uint i = 0; i < edge_count; i++) {
    struct _cl_work_graph_edge_formosa *edge = ordered_edges[i];
    int src_idx = pocl_formosa_find_node_index_by_ptr(ordered_nodes, node_count,
                                                      edge->src_node);
    int dst_idx = pocl_formosa_find_node_index_by_ptr(ordered_nodes, node_count,
                                                      edge->dst_node);
    if (src_idx < 0 || dst_idx < 0) {
      POCL_MSG_ERR("formosa: graph edge %u references a node outside graph\n",
                   edge->edge_id);
      goto CLEANUP;
    }
    if (src_idx == dst_idx) {
      POCL_MSG_ERR("formosa: graph edge %u is a self-edge\n", edge->edge_id);
      goto CLEANUP;
    }
    if (edge->dst_node == root_node) {
      POCL_MSG_ERR(
          "formosa: edges into the root node are unsupported while "
          "root input records are aliased\n");
      goto CLEANUP;
    }
    for (cl_uint j = i + 1; j < edge_count; j++) {
      if (edge->edge_id == ordered_edges[j]->edge_id) {
        POCL_MSG_ERR("formosa: duplicate graph edge id %u\n", edge->edge_id);
        goto CLEANUP;
      }
    }

    uint32_t record_size = (uint32_t)edge->properties.record_size;
    if (record_size == 0)
      record_size = (uint32_t)edge->dst_node->properties.input_record_size;
    if (record_size == 0 ||
        record_size != edge->dst_node->properties.input_record_size) {
      POCL_MSG_ERR(
          "formosa: graph edge %u record size %u does not match "
          "destination node %u input size %zu\n",
          edge->edge_id, record_size, edge->dst_node->node_id,
          edge->dst_node->properties.input_record_size);
      goto CLEANUP;
    }

    edge_descs[i].edge_id = edge->edge_id;
    edge_descs[i].src_node_id = edge->src_node->node_id;
    edge_descs[i].dst_node_id = edge->dst_node->node_id;
    edge_descs[i].record_size = record_size;
    edge_descs[i].queue_capacity = edge->properties.queue_capacity;
    edge_descs[i].max_records_per_src_record =
        edge->properties.max_records_per_src_record;
    if (edge_descs[i].max_records_per_src_record == 0) {
      /* Admission interprets this as a per-source-input-record emit bound.
         Producers that can emit more than one output record per input must
         specify an explicit bound. */
      edge_descs[i].max_records_per_src_record = 1;
    }

    node_unresolved_in_edges[dst_idx]++;
  }

  int cycle_check = pocl_formosa_graph_has_cycle(ordered_nodes, node_count,
                                                 ordered_edges, edge_count);
  if (cycle_check < 0) {
    POCL_MSG_ERR("formosa: out of host memory during DAG validation\n");
    goto CLEANUP;
  }
  if (cycle_check) {
    POCL_MSG_ERR(
        "formosa: graph contains a cycle; DAGs only are supported for now\n");
    goto CLEANUP;
  }

  const cl_uint root_idx = 0;

  if (root_inputs[0].record_count > node_queue_caps[root_idx])
    node_queue_caps[root_idx] = root_inputs[0].record_count;

  /* Infer destination queue storage. An explicit edge queue_capacity wins. If
     the app leaves it at 0, derive a conservative capacity from the source
     queue capacity and the per-source-record emit bound. This keeps queue
     storage policy in PoCL instead of tying it to dispatch batch size. */
  cl_uint resolved_edges = 0;
  for (cl_uint pass = 0; pass < node_count && resolved_edges < edge_count;
       pass++) {
    int made_progress = 0;
    for (cl_uint i = 0; i < edge_count; i++) {
      if (edge_capacity_resolved[i]) continue;

      struct _cl_work_graph_edge_formosa *edge = ordered_edges[i];
      int src_idx = pocl_formosa_find_node_index_by_ptr(
          ordered_nodes, node_count, edge->src_node);
      int dst_idx = pocl_formosa_find_node_index_by_ptr(
          ordered_nodes, node_count, edge->dst_node);

      if (src_idx != (int)root_idx && node_unresolved_in_edges[src_idx] != 0)
        continue;

      uint32_t queue_capacity = edge_descs[i].queue_capacity;
      if (queue_capacity == 0) {
        if (node_queue_caps[src_idx] == 0) continue;
        if (pocl_formosa_mul_u32_overflow(
                node_queue_caps[src_idx],
                edge_descs[i].max_records_per_src_record, &queue_capacity)) {
          POCL_MSG_ERR(
              "formosa: graph edge %u inferred queue capacity overflows\n",
              edge->edge_id);
          goto CLEANUP;
        }
        if (queue_capacity == 0)
          queue_capacity =
              pocl_formosa_default_queue_capacity(root_inputs[0].record_count);
      }

      if (UINT32_MAX - node_queue_caps[dst_idx] < queue_capacity) {
        POCL_MSG_ERR("formosa: graph node %u queue capacity overflow\n",
                     edge->dst_node->node_id);
        goto CLEANUP;
      }

      edge_descs[i].queue_capacity = queue_capacity;
      node_queue_caps[dst_idx] += queue_capacity;
      edge_capacity_resolved[i] = 1;
      node_unresolved_in_edges[dst_idx]--;
      resolved_edges++;
      made_progress = 1;
    }

    if (!made_progress) break;
  }

  if (resolved_edges != edge_count) {
    POCL_MSG_ERR(
        "formosa: failed to infer queue capacities; provide explicit edge "
        "queue_capacity for disconnected or non-root-reachable graph edges\n");
    goto CLEANUP;
  }

  formosa_buffer_data_t *root_buf_data = NULL;
  err = pocl_formosa_prepare_mem_arg(device, root_inputs[0].records,
                                     &root_buf_data);
  if (err != CL_SUCCESS) {
    POCL_MSG_ERR("formosa: failed to prepare root input memory\n");
    goto CLEANUP;
  }

  uint32_t root_record_size =
      (uint32_t)ordered_nodes[root_idx]->properties.input_record_size;
  if (root_inputs[0].record_count > 0 &&
      root_inputs[0].record_stride != root_record_size) {
    POCL_MSG_ERR(
        "formosa: ring queue root seeding requires packed root records "
        "(stride=%zu record_size=%u)\n",
        root_inputs[0].record_stride, root_record_size);
    goto CLEANUP;
  }

  for (cl_uint i = 0; i < node_count; i++) {
    if (node_queue_caps[i] == 0) {
      node_queue_caps[i] =
          pocl_formosa_default_queue_capacity(root_inputs[0].record_count);
    }
  }

  if (bg->dev_graph_status == 0 &&
      fsa_malloc((void **)&bg->dev_graph_status, sizeof(GraphStatus))) {
    POCL_MSG_ERR("formosa: failed to allocate graph status\n");
    goto CLEANUP;
  }
  if (bg->dev_graph_desc == 0)
    fsa_malloc((void **)&bg->dev_graph_desc, sizeof(GraphDescriptor));
  if (bg->dev_runtime_pool == 0)
    fsa_malloc((void **)&bg->dev_runtime_pool, sizeof(GraphRuntimePool));

  if (bg->dev_node_descs == 0 || bg->node_desc_capacity < node_count) {
    if (bg->dev_node_descs) fsa_free((void *)bg->dev_node_descs);
    fsa_malloc((void **)&bg->dev_node_descs,
               sizeof(NodeDescriptor) * node_count);
    bg->node_desc_capacity = node_count;
  }

  if (edge_count == 0) {
    if (bg->dev_edge_descs) fsa_free((void *)bg->dev_edge_descs);
    bg->dev_edge_descs = 0;
    bg->edge_desc_capacity = 0;
  } else if (bg->dev_edge_descs == 0 || bg->edge_desc_capacity < edge_count) {
    if (bg->dev_edge_descs) fsa_free((void *)bg->dev_edge_descs);
    fsa_malloc((void **)&bg->dev_edge_descs,
               sizeof(EdgeDescriptor) * edge_count);
    bg->edge_desc_capacity = edge_count;
  }

  if (bg->dev_node_queues == 0 || bg->node_queue_capacity < node_count ||
      bg->dev_node_queue_records == NULL ||
      bg->dev_node_queue_ready_sequences == NULL ||
      bg->node_queue_record_sizes == NULL ||
      bg->node_queue_ready_sequence_sizes == NULL) {
    if (bg->dev_node_queues) fsa_free((void *)bg->dev_node_queues);
    pocl_formosa_free_node_queue_storage(bg);
    bg->dev_node_queues = 0;
    if (fsa_malloc((void **)&bg->dev_node_queues,
                   sizeof(NodeInputQueue) * node_count)) {
      POCL_MSG_ERR("formosa: failed to allocate node input queues\n");
      goto CLEANUP;
    }
    bg->dev_node_queue_records =
        (uint64_t *)calloc(node_count, sizeof(uint64_t));
    bg->dev_node_queue_ready_sequences =
        (uint64_t *)calloc(node_count, sizeof(uint64_t));
    bg->node_queue_record_sizes = (size_t *)calloc(node_count, sizeof(size_t));
    bg->node_queue_ready_sequence_sizes =
        (size_t *)calloc(node_count, sizeof(size_t));
    if (bg->dev_node_queue_records == NULL ||
        bg->dev_node_queue_ready_sequences == NULL ||
        bg->node_queue_record_sizes == NULL ||
        bg->node_queue_ready_sequence_sizes == NULL) {
      POCL_MSG_ERR("formosa: out of host memory for node queue bookkeeping\n");
      goto CLEANUP;
    }
    bg->node_queue_capacity = node_count;
  }

  if (bg->dev_node_states == 0 || bg->node_state_capacity < node_count) {
    if (bg->dev_node_states) fsa_free((void *)bg->dev_node_states);
    fsa_malloc((void **)&bg->dev_node_states,
               sizeof(NodeRuntimeState) * node_count);
    bg->node_state_capacity = node_count;
  }

  GraphStatus initial_status = {0};
  initial_status.state = kGraphStatusIdle;
  if (fsa_copy_to_dev(bg->dev_graph_status, &initial_status,
                      sizeof(initial_status))) {
    POCL_MSG_ERR("formosa: failed to initialize graph status\n");
    goto CLEANUP;
  }

  for (cl_uint i = 0; i < node_count; i++) {
    struct _cl_work_graph_node_formosa *node = ordered_nodes[i];
    NodeDescriptor *nd = &node_descs[i];

    char sz_program_fsabin[POCL_MAX_PATHNAME_LENGTH];
    pocl_fsa_get_elf_name(node->kernel->program, 0, sz_program_fsabin);

    uint64_t dev_kernel_addr = 0;
    pocl_fsa_upload_kernel(sz_program_fsabin, dd, &dev_kernel_addr);

    uint64_t entry_pc =
        pocl_fsa_get_symbol_pc(sz_program_fsabin, "_start") + dev_kernel_addr;
    char trampoline_name[256];
    snprintf(trampoline_name, sizeof(trampoline_name), "%s_trampolined",
             node->kernel->name);
    uint64_t trampoline_pc =
        pocl_fsa_get_symbol_pc(sz_program_fsabin, trampoline_name) +
        dev_kernel_addr;

    nd->node_id = node->node_id;
    nd->flags = node->properties.flags;
    nd->launch_mode = node->properties.launch_mode;
    nd->input_record_size = node->properties.input_record_size;
    nd->max_input_records_per_dispatch =
        node->properties.max_input_records_per_dispatch;
    nd->max_active_dispatches = node->properties.max_active_dispatches;
    nd->work_items_per_input_record =
        node->properties.work_items_per_input_record != 0
            ? node->properties.work_items_per_input_record
            : 1;
    nd->kernel_object = entry_pc;
    nd->kernel_trampoline = trampoline_pc;
    nd->work_dim = node->work_dim;
    for (int j = 0; j < 3; j++) {
      nd->global_work_offset[j] = node->global_work_offset[j];
      nd->global_work_size[j] = node->global_work_size[j];
      nd->local_work_size[j] = node->local_work_size[j];
      if (j < (int)node->work_dim && nd->local_work_size[j] == 0)
        nd->local_work_size[j] = 1;
    }

    uint64_t dev_kargs_addr = 0;
    uint32_t kargs_size = 0;
    uint64_t local_mem_size = 0;

    /* Capture the kernel's current clSetKernelArg state for this launch. */
    err = pocl_formosa_pack_kernel_args(
        device, node->kernel, node->node_id, node->properties.flags,
        &dev_kargs_addr, &kargs_size, &local_mem_size, &cache_ranges,
        &cache_range_count, &cache_range_capacity);
    if (err != CL_SUCCESS) {
      POCL_MSG_ERR("formosa: failed to pack kernel args for graph node %u\n",
                   node->node_id);
      goto CLEANUP;
    }
    if (kargs_size > 256) {
      POCL_MSG_ERR(
          "formosa: graph node %u static kernargs exceed static slot size "
          "(%u > 256)\n",
          node->node_id, kargs_size);
      fsa_free((void *)dev_kargs_addr);
      goto CLEANUP;
    }

    launch_static_kargs[i] = dev_kargs_addr;
    launch_static_karg_sizes[i] = kargs_size;
    launch_local_mem_sizes[i] = local_mem_size;

    nd->static_kernarg_addr = launch_static_kargs[i];
    nd->static_kernarg_size = launch_static_karg_sizes[i];
    nd->local_mem_size = launch_local_mem_sizes[i];
  }

  fsa_copy_to_dev(bg->dev_node_descs, node_descs,
                  sizeof(NodeDescriptor) * node_count);

  if (edge_count > 0)
    fsa_copy_to_dev(bg->dev_edge_descs, edge_descs,
                    sizeof(EdgeDescriptor) * edge_count);

  if (cache_range_count > 0) {
    size_t cache_ranges_size =
        (size_t)cache_range_count * sizeof(GraphCacheRange);
    if (fsa_malloc((void **)&dev_cache_ranges, cache_ranges_size) ||
        fsa_copy_to_dev(dev_cache_ranges, cache_ranges, cache_ranges_size)) {
      POCL_MSG_ERR("formosa: failed to upload graph cache ranges\n");
      goto CLEANUP;
    }
  }

  for (cl_uint i = 0; i < node_count; i++) {
    size_t input_record_size = ordered_nodes[i]->properties.input_record_size;
    if (input_record_size > UINT32_MAX - 15u) {
      POCL_MSG_ERR("formosa: graph node %u record size is too large\n",
                   ordered_nodes[i]->node_id);
      goto CLEANUP;
    }
    uint32_t record_size = (uint32_t)input_record_size;
    uint32_t record_stride = (record_size + 15u) & ~15u;
    if ((int)i == root_idx) record_stride = record_size;
    uint32_t capacity = node_queue_caps[i];
    if (record_stride != 0 && capacity > SIZE_MAX / record_stride) {
      POCL_MSG_ERR(
          "formosa: graph node %u queue records allocation overflows size_t\n",
          ordered_nodes[i]->node_id);
      goto CLEANUP;
    }
    size_t records_size = (size_t)record_stride * capacity;
    size_t ready_sequences_size = sizeof(uint32_t) * capacity;
    uint32_t *ready_sequences =
        (uint32_t *)calloc(capacity, sizeof(*ready_sequences));
    if (ready_sequences == NULL) {
      POCL_MSG_ERR("formosa: out of host memory for node ready sequences\n");
      goto CLEANUP;
    }

    node_queues[i].reserve_tail = 0;
    node_queues[i].ready_tail = 0;
    node_queues[i].consumed_head = 0;
    node_queues[i].dispatch_head = 0;
    node_queues[i].admission_reserved = 0;
    node_queues[i].capacity = capacity;
    node_queues[i].record_size = record_size;
    node_queues[i].record_stride = record_stride;

    if ((int)i == root_idx) {
      if (bg->dev_node_queue_records[i]) {
        fsa_free((void *)bg->dev_node_queue_records[i]);
        bg->dev_node_queue_records[i] = 0;
        bg->node_queue_record_sizes[i] = 0;
      }
      node_queues[i].reserve_tail = root_inputs[0].record_count;
      node_queues[i].ready_tail = root_inputs[0].record_count;
      node_queues[i].records_addr =
          root_buf_data->buf_address + root_inputs[0].records_offset;
      for (cl_uint j = 0; j < root_inputs[0].record_count; j++) {
        ready_sequences[j] = j + 1;
      }
    } else {
      if (bg->dev_node_queue_records[i] == 0 ||
          bg->node_queue_record_sizes[i] < records_size) {
        if (bg->dev_node_queue_records[i])
          fsa_free((void *)bg->dev_node_queue_records[i]);
        bg->dev_node_queue_records[i] = 0;
        bg->node_queue_record_sizes[i] = 0;
        if (fsa_malloc_noncache((void **)&bg->dev_node_queue_records[i],
                                records_size)) {
          POCL_MSG_ERR(
              "formosa: failed to allocate non-cacheable node queue records "
              "node=%u cap=%u record_size=%u bytes=%zu\n",
              ordered_nodes[i]->node_id, capacity, record_size, records_size);
          free(ready_sequences);
          goto CLEANUP;
        }
        bg->node_queue_record_sizes[i] = records_size;
      }

      node_queues[i].records_addr = bg->dev_node_queue_records[i];
    }

    if (bg->dev_node_queue_ready_sequences[i] == 0 ||
        bg->node_queue_ready_sequence_sizes[i] < ready_sequences_size) {
      if (bg->dev_node_queue_ready_sequences[i]) {
        fsa_free((void *)bg->dev_node_queue_ready_sequences[i]);
        bg->dev_node_queue_ready_sequences[i] = 0;
        bg->node_queue_ready_sequence_sizes[i] = 0;
      }
      /* Firmware polls ready_sequence while producers may still be active, so
       * these per-slot completion markers must bypass the SM D-cache.
       */
      if (fsa_malloc_noncache((void **)&bg->dev_node_queue_ready_sequences[i],
                              ready_sequences_size)) {
        POCL_MSG_ERR(
            "formosa: failed to allocate non-cacheable node ready sequences\n");
        free(ready_sequences);
        goto CLEANUP;
      }
      bg->node_queue_ready_sequence_sizes[i] = ready_sequences_size;
    }

    node_queues[i].ready_sequence_addr = bg->dev_node_queue_ready_sequences[i];
    if (fsa_copy_to_dev(node_queues[i].ready_sequence_addr, ready_sequences,
                        ready_sequences_size)) {
      POCL_MSG_ERR("formosa: failed to initialize node ready sequences\n");
      free(ready_sequences);
      goto CLEANUP;
    }
    free(ready_sequences);
  }

  fsa_copy_to_dev(bg->dev_node_queues, node_queues,
                  sizeof(NodeInputQueue) * node_count);
  fsa_copy_to_dev(bg->dev_node_states, node_states,
                  sizeof(NodeRuntimeState) * node_count);

  /* Fill Graph Descriptor */
  GraphDescriptor gd = {0};
  gd.version = FORMOSA_WORK_GRAPH_ABI_VERSION;
  gd.node_count = node_count;
  gd.edge_count = edge_count;
  gd.node_desc_addr = bg->dev_node_descs;
  gd.edge_desc_addr = bg->dev_edge_descs;
  fsa_copy_to_dev(bg->dev_graph_desc, &gd, sizeof(GraphDescriptor));

  /* Fill Root Input Descriptor */
  RootInputDescriptor rid = {0};
  rid.target_node_id = root_inputs[0].target_node_id;
  rid.record_count = root_inputs[0].record_count;
  rid.records_addr = root_buf_data->buf_address;
  rid.records_offset = root_inputs[0].records_offset;
  rid.record_stride = root_inputs[0].record_stride;

  fsa_malloc((void **)&dev_root_desc, sizeof(RootInputDescriptor));
  fsa_copy_to_dev(dev_root_desc, &rid, sizeof(RootInputDescriptor));

  /* Fill Runtime Pool */
  cl_uint dispatch_slot_count = node_count + FORMOSA_WG_EXTRA_DISPATCH_SLOTS;
  if (dispatch_slot_count > FORMOSA_WG_MAX_DISPATCH_SLOTS)
    dispatch_slot_count = FORMOSA_WG_MAX_DISPATCH_SLOTS;
  pool.ctx_stride = sizeof(WorkGraphNodeContext);
  pool.ctx_slot_count = dispatch_slot_count;
  fsa_malloc((void **)&pool.ctx_pool_base,
             pool.ctx_stride * pool.ctx_slot_count);
  pool.kernarg_stride = 256;
  pool.kernarg_slot_count = dispatch_slot_count;
  fsa_malloc((void **)&pool.kernarg_pool_base,
             pool.kernarg_stride * pool.kernarg_slot_count);
  pool.graph_desc_addr = bg->dev_graph_desc;
  pool.node_queue_addr = bg->dev_node_queues;
  pool.node_queue_stride = sizeof(NodeInputQueue);
  pool.node_queue_count = node_count;
  pool.node_state_addr = bg->dev_node_states;
  pool.node_state_stride = sizeof(NodeRuntimeState);
  pool.node_state_count = node_count;
  pool.cache_range_count = cache_range_count;
  pool.cache_range_addr = dev_cache_ranges;
  fsa_copy_to_dev(bg->dev_runtime_pool, &pool, sizeof(GraphRuntimePool));

  /* Launch */
  uintptr_t completion_signal = 0;

  int launch_rc = fsa_cmd_launch_graph(
      0, num_root_inputs, bg->dev_graph_desc, bg->dev_runtime_pool,
      bg->dev_graph_status, dev_root_desc,
      rid.records_addr + rid.records_offset, &completion_signal);

  /* Wait synchronously for graph completion. */
  if (launch_rc == 0) {
    pocl_fsa_wait_ack(dd, completion_signal, 0);
  } else {
    POCL_MSG_ERR("formosa: graph launch failed\n");
  }

  /* Clean up per-launch allocations. */
  if (dev_root_desc) {
    fsa_free((void *)dev_root_desc);
    dev_root_desc = 0;
  }
  if (pool.ctx_pool_base) {
    fsa_free((void *)pool.ctx_pool_base);
    pool.ctx_pool_base = 0;
  }
  if (pool.kernarg_pool_base) {
    fsa_free((void *)pool.kernarg_pool_base);
    pool.kernarg_pool_base = 0;
  }
  if (dev_cache_ranges) {
    fsa_free((void *)dev_cache_ranges);
    dev_cache_ranges = 0;
  }

CLEANUP:
  if (launch_static_kargs) {
    for (cl_uint i = 0; i < node_count; i++) {
      if (launch_static_kargs[i]) fsa_free((void *)launch_static_kargs[i]);
    }
  }
  if (dev_root_desc) fsa_free((void *)dev_root_desc);
  if (pool.ctx_pool_base) fsa_free((void *)pool.ctx_pool_base);
  if (pool.kernarg_pool_base) fsa_free((void *)pool.kernarg_pool_base);
  if (dev_cache_ranges) fsa_free((void *)dev_cache_ranges);
  free(listed_nodes);
  free(ordered_nodes);
  free(ordered_edges);
  free(node_queue_caps);
  free(node_descs);
  free(edge_descs);
  free(node_queues);
  free(node_states);
  free(edge_capacity_resolved);
  free(node_unresolved_in_edges);
  free(launch_static_kargs);
  free(launch_static_karg_sizes);
  free(launch_local_mem_sizes);
  free(cache_ranges);
}

static cl_int pocl_formosa_read_graph_status(
    struct pocl_formosa_work_graph_data *bg, GraphStatus *status) {
  if (bg == NULL || bg->dev_graph_status == 0) return CL_INVALID_VALUE;
  if (fsa_copy_from_dev(bg->dev_graph_status, status, sizeof(*status)) != 0)
    return CL_OUT_OF_RESOURCES;
  return CL_SUCCESS;
}

cl_int pocl_formosa_get_work_graph_info(cl_work_graph_formosa graph,
                                        cl_uint param, size_t size, void *value,
                                        size_t *size_ret) {
  if (graph == NULL) return CL_INVALID_VALUE;

  struct pocl_formosa_work_graph_data *bg =
      (struct pocl_formosa_work_graph_data *)graph->backend_data;

  if (param == CL_GRAPH_INFO_STATUS_FORMOSA ||
      param == CL_GRAPH_INFO_LAST_ERROR_FORMOSA ||
      param == CL_GRAPH_INFO_MAX_ACTIVE_DISPATCHES_FORMOSA ||
      param == CL_GRAPH_INFO_MAX_NODE_ACTIVE_DISPATCHES_FORMOSA) {
    GraphStatus status = {0};
    cl_int err = pocl_formosa_read_graph_status(bg, &status);
    if (err != CL_SUCCESS) return err;

    if (value) {
      if (size < sizeof(cl_uint)) return CL_INVALID_VALUE;
      if (param == CL_GRAPH_INFO_STATUS_FORMOSA) {
        *(cl_uint *)value = (cl_uint)status.state;
      } else if (param == CL_GRAPH_INFO_LAST_ERROR_FORMOSA) {
        *(cl_uint *)value = (cl_uint)status.error_code;
      } else if (param == CL_GRAPH_INFO_MAX_NODE_ACTIVE_DISPATCHES_FORMOSA) {
        *(cl_uint *)value = status.max_node_active_dispatches;
      } else {
        *(cl_uint *)value = status.max_active_dispatches;
      }
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
    if (bg->dev_edge_descs) fsa_free((void *)bg->dev_edge_descs);
    if (bg->dev_node_queues) fsa_free((void *)bg->dev_node_queues);
    if (bg->dev_node_states) fsa_free((void *)bg->dev_node_states);
    pocl_formosa_free_node_queue_storage(bg);
    if (bg->dev_runtime_pool) fsa_free((void *)bg->dev_runtime_pool);
    free(bg);
  }

  return CL_SUCCESS;
}
