#include "wg_info.h"

struct EdgeDescriptor {
  uint32_t edge_id;
  uint32_t src_node_id;
  uint32_t dst_node_id;
  uint32_t record_size;

  uint32_t queue_capacity;
  uint32_t reserved0;
};

struct NodeInputQueue {
  volatile uint32_t tail;
  volatile uint32_t consumed_count;
  uint32_t capacity;
  uint32_t record_size;

  uint64_t records_addr;
};

struct GraphDescriptor {
  uint32_t version;
  uint32_t flags;

  uint32_t node_count;
  uint32_t edge_count;

  uint64_t node_desc_addr;
  uint64_t edge_desc_addr;
};

struct GraphRuntimePool {
  uint64_t ctx_pool_base;
  uint32_t ctx_stride;
  uint32_t ctx_slot_count;

  uint64_t kernarg_pool_base;
  uint32_t kernarg_stride;
  uint32_t kernarg_slot_count;

  uint64_t graph_desc_addr;

  uint64_t node_queue_addr;
  uint32_t node_queue_stride;
  uint32_t node_queue_count;

  uint64_t node_state_addr;
  uint32_t node_state_stride;
  uint32_t node_state_count;
};

struct NodeDescriptor {
  uint32_t node_id;
  uint32_t flags;
  uint32_t launch_mode;
  uint32_t input_record_size;

  uint64_t kernel_object;
  uint64_t kernel_trampoline;

  uint32_t work_dim;
  uint32_t reserved0;

  uint64_t global_work_offset[3];
  uint64_t global_work_size[3];
  uint64_t local_work_size[3];

  uint64_t local_mem_size;
  uint64_t static_kernarg_addr;
  uint32_t static_kernarg_size;
  uint32_t reserved1;
};

uint formosa_get_record_count(void) {
  struct WGInfo *info = get_wg_info();
  __global struct WorkGraphNodeContext *ctx =
      (__global struct WorkGraphNodeContext *)(uintptr_t)info->work_graph_ctx;
  if (ctx == NULL)
    return 0;
  return ctx->input_record_count;
}

int formosa_get_record(uint index, __private void *record_out,
                       size_t record_size) {
  struct WGInfo *info = get_wg_info();
  __global struct WorkGraphNodeContext *ctx =
      (__global struct WorkGraphNodeContext *)(uintptr_t)info->work_graph_ctx;

  if (ctx == NULL)
    return -1;
  if (index >= ctx->input_record_count)
    return -1;
  if (record_size != (size_t)ctx->input_record_size)
    return -1;

  size_t stride = (size_t)ctx->input_record_size;
  __global uchar *src = (__global uchar *)(uintptr_t)ctx->input_records +
                        ((size_t)index * stride);
  __private uchar *dst = (__private uchar *)record_out;

  for (size_t i = 0; i < record_size; i++) {
    dst[i] = src[i];
  }

  return 0;
}

int formosa_emit(uint edge_id, const __private void *record) {
  if (record == NULL)
    return -1;

  struct WGInfo *info = get_wg_info();
  __global struct WorkGraphNodeContext *ctx =
      (__global struct WorkGraphNodeContext *)(uintptr_t)info->work_graph_ctx;
  if (ctx == NULL)
    return -2;
  if (ctx->edge_count == 0 || ctx->edge_table == 0)
    return -3;
  if (ctx->graph_runtime == 0)
    return -4;

  __global struct GraphRuntimePool *pool =
      (__global struct GraphRuntimePool *)(uintptr_t)ctx->graph_runtime;
  if (pool == NULL || pool->graph_desc_addr == 0)
    return -5;
  if (pool->node_queue_addr == 0 ||
      pool->node_queue_stride < sizeof(struct NodeInputQueue) ||
      pool->node_queue_count == 0) {
    return -6;
  }

  __global struct GraphDescriptor *graph =
      (__global struct GraphDescriptor *)(uintptr_t)pool->graph_desc_addr;
  if (graph == NULL || graph->node_count == 0 || graph->node_desc_addr == 0) {
    return -7;
  }

  __global struct EdgeDescriptor *edges =
      (__global struct EdgeDescriptor *)(uintptr_t)ctx->edge_table;

  __global struct EdgeDescriptor *edge = 0;
  for (uint i = 0; i < ctx->edge_count; i++) {
    if (edges[i].edge_id == edge_id) {
      edge = &edges[i];
      break;
    }
  }

  if (edge == 0)
    return -8;
  if (edge->src_node_id != ctx->node_id)
    return -9;
  if (edge->record_size == 0 || edge->queue_capacity == 0)
    return -10;

  __global struct NodeDescriptor *nodes =
      (__global struct NodeDescriptor *)(uintptr_t)graph->node_desc_addr;
  uint dst_index = graph->node_count;
  for (uint i = 0; i < graph->node_count; i++) {
    if (nodes[i].node_id == edge->dst_node_id) {
      dst_index = i;
      break;
    }
  }

  if (dst_index >= graph->node_count)
    return -11;
  if (dst_index >= pool->node_queue_count)
    return -12;

  __global uchar *queue_base =
      (__global uchar *)(uintptr_t)pool->node_queue_addr;
  __global struct NodeInputQueue *queue =
      (__global struct NodeInputQueue *)(queue_base +
                                         ((size_t)dst_index *
                                          (size_t)pool->node_queue_stride));

  uint record_size = queue->record_size;
  if (queue->capacity == 0 || record_size == 0 || queue->records_addr == 0) {
    return -13;
  }
  if (record_size != edge->record_size)
    return -14;

  volatile __global uint *tail = (volatile __global uint *)&queue->tail;
  uint slot = atomic_inc(tail);
  if (slot >= queue->capacity) {
    /* Formosa does not currently provide the OpenCL compare-and-swap builtin
       in the kernel library. Undo the supported fetch-add reservation on
       overflow so the final retired queue tail is not left past capacity in
       Phase 3. The final protocol still needs bounded reservation plus ready
       publication. */
    atomic_dec(tail);
    return -15;
  }

  __global uchar *dst = (__global uchar *)(uintptr_t)queue->records_addr +
                        ((size_t)slot * record_size);
  const __private uchar *src = (const __private uchar *)record;
  for (uint i = 0; i < record_size; i++) {
    dst[i] = src[i];
  }

  /* Consumers are scheduled by firmware only after the producer node retires,
     so this path relies on retire-driven visibility. */
  return 0;
}
