#include "wg_info.h"

#define FORMOSA_WG_SUCCESS 0
#define FORMOSA_WG_ERR_NULL_RECORD -1
#define FORMOSA_WG_ERR_NULL_CONTEXT -2
#define FORMOSA_WG_ERR_NO_EDGE_TABLE -3
#define FORMOSA_WG_ERR_NO_RUNTIME -4
#define FORMOSA_WG_ERR_BAD_RUNTIME_POOL -5
#define FORMOSA_WG_ERR_BAD_NODE_QUEUE_TABLE -6
#define FORMOSA_WG_ERR_BAD_GRAPH_DESC -7
#define FORMOSA_WG_ERR_EDGE_NOT_FOUND -8
#define FORMOSA_WG_ERR_WRONG_SOURCE_NODE -9
#define FORMOSA_WG_ERR_BAD_EDGE_DESC -10
#define FORMOSA_WG_ERR_DST_NODE_NOT_FOUND -11
#define FORMOSA_WG_ERR_DST_QUEUE_OUT_OF_RANGE -12
#define FORMOSA_WG_ERR_BAD_QUEUE -13
#define FORMOSA_WG_ERR_RECORD_SIZE_MISMATCH -14
#define FORMOSA_WG_ERR_QUEUE_OVERFLOW -15
#define FORMOSA_WG_ERR_BAD_READY_SEQUENCE -16
#define FORMOSA_WG_ERR_RECORD_OUT_OF_RANGE -17
#define FORMOSA_WORK_GRAPH_ABI_VERSION 2

struct EdgeDescriptor {
  uint32_t edge_id;
  uint32_t src_node_id;
  uint32_t dst_node_id;
  uint32_t record_size;

  uint32_t queue_capacity;
  uint32_t max_records_per_src_record;
};

struct NodeInputQueue {
  volatile uint32_t reserve_tail;
  volatile uint32_t ready_tail;
  volatile uint32_t consumed_head;
  volatile uint32_t dispatch_head;
  volatile uint32_t admission_reserved;

  uint32_t capacity;
  uint32_t record_size;
  uint32_t record_stride;

  uint64_t records_addr;
  uint64_t ready_sequence_addr;
};

_Static_assert(sizeof(struct NodeInputQueue) == 48,
               "NodeInputQueue size mismatch");
_Static_assert(offsetof(struct NodeInputQueue, record_stride) == 28,
               "NodeInputQueue.record_stride offset mismatch");

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
  uint32_t cache_range_count;

  uint64_t cache_range_addr;
};

_Static_assert(sizeof(struct GraphRuntimePool) == 88,
               "GraphRuntimePool size mismatch");
_Static_assert(offsetof(struct GraphRuntimePool, cache_range_count) == 72,
               "GraphRuntimePool.cache_range_count offset mismatch");
_Static_assert(offsetof(struct GraphRuntimePool, cache_range_addr) == 80,
               "GraphRuntimePool.cache_range_addr offset mismatch");

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
  uint32_t max_input_records_per_dispatch;
  uint32_t max_active_dispatches;
  uint32_t work_items_per_input_record;
};

static inline void formosa_wg_publish_ready(volatile __global uint *ready_slot,
                                            uint value) {
  *ready_slot = value;
}

static inline uint formosa_wg_reserve_ticket(volatile __global uint *tail) {
  uint old;
  uint one = 1;
  __asm__ volatile("amoadd.w.aqrl %0, %2, (%1)"
                   : "=r"(old)
                   : "r"(tail), "r"(one)
                   : "memory");
  return old;
}

uint formosa_get_record_count(void) {
  struct WGInfo *info = get_wg_info();
  __global struct WorkGraphNodeContext *ctx =
      (__global struct WorkGraphNodeContext *)(uintptr_t)info->work_graph_ctx;
  if (ctx == NULL)
    return 0;
  return ctx->input_record_count;
}

uint formosa_get_record_work_item_count(void) {
  struct WGInfo *info = get_wg_info();
  __global struct WorkGraphNodeContext *ctx =
      (__global struct WorkGraphNodeContext *)(uintptr_t)info->work_graph_ctx;
  if (ctx == NULL || ctx->work_items_per_input_record == 0)
    return 1;
  return ctx->work_items_per_input_record;
}

uint formosa_get_current_record_index(void) {
  uint expansion = formosa_get_record_work_item_count();
  return (uint)get_global_id(0) / expansion;
}

uint formosa_get_current_record_work_item_id(void) {
  uint expansion = formosa_get_record_work_item_count();
  return (uint)get_global_id(0) % expansion;
}

int formosa_get_record(uint index, __private void *record_out,
                       size_t record_size) {
  struct WGInfo *info = get_wg_info();
  __global struct WorkGraphNodeContext *ctx =
      (__global struct WorkGraphNodeContext *)(uintptr_t)info->work_graph_ctx;

  if (record_out == NULL)
    return FORMOSA_WG_ERR_NULL_RECORD;
  if (ctx == NULL)
    return FORMOSA_WG_ERR_NULL_CONTEXT;
  if (index >= ctx->input_record_count)
    return FORMOSA_WG_ERR_RECORD_OUT_OF_RANGE;
  if (record_size != (size_t)ctx->input_record_size)
    return FORMOSA_WG_ERR_RECORD_SIZE_MISMATCH;

  size_t stride = ctx->input_record_stride != 0
                      ? (size_t)ctx->input_record_stride
                      : (size_t)ctx->input_record_size;
  if (stride < record_size)
    return FORMOSA_WG_ERR_RECORD_SIZE_MISMATCH;
  __global uchar *src = (__global uchar *)(uintptr_t)ctx->input_records +
                        ((size_t)index * stride);
  __private uchar *dst = (__private uchar *)record_out;

  if ((((uintptr_t)src | (uintptr_t)dst | record_size) & 3u) == 0) {
    __global uint *src_words = (__global uint *)src;
    __private uint *dst_words = (__private uint *)dst;
    for (size_t i = 0; i < record_size / sizeof(uint); i++) {
      dst_words[i] = src_words[i];
    }
  } else {
    for (size_t i = 0; i < record_size; i++) {
      dst[i] = src[i];
    }
  }

  return FORMOSA_WG_SUCCESS;
}

int formosa_emit(uint edge_id, const __private void *record) {
  if (record == NULL)
    return FORMOSA_WG_ERR_NULL_RECORD;

  struct WGInfo *info = get_wg_info();
  __global struct WorkGraphNodeContext *ctx =
      (__global struct WorkGraphNodeContext *)(uintptr_t)info->work_graph_ctx;
  if (ctx == NULL)
    return FORMOSA_WG_ERR_NULL_CONTEXT;
  if (ctx->edge_count == 0 || ctx->edge_table == 0)
    return FORMOSA_WG_ERR_NO_EDGE_TABLE;
  if (ctx->graph_runtime == 0)
    return FORMOSA_WG_ERR_NO_RUNTIME;

  __global struct GraphRuntimePool *pool =
      (__global struct GraphRuntimePool *)(uintptr_t)ctx->graph_runtime;
  if (pool == NULL || pool->graph_desc_addr == 0)
    return FORMOSA_WG_ERR_BAD_RUNTIME_POOL;
  if (pool->node_queue_addr == 0 ||
      pool->node_queue_stride < sizeof(struct NodeInputQueue) ||
      pool->node_queue_count == 0) {
    return FORMOSA_WG_ERR_BAD_NODE_QUEUE_TABLE;
  }

  __global struct GraphDescriptor *graph =
      (__global struct GraphDescriptor *)(uintptr_t)pool->graph_desc_addr;
  if (graph == NULL || graph->version != FORMOSA_WORK_GRAPH_ABI_VERSION ||
      graph->node_count == 0 || graph->node_desc_addr == 0) {
    return FORMOSA_WG_ERR_BAD_GRAPH_DESC;
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
    return FORMOSA_WG_ERR_EDGE_NOT_FOUND;
  if (edge->src_node_id != ctx->node_id)
    return FORMOSA_WG_ERR_WRONG_SOURCE_NODE;
  if (edge->record_size == 0 || edge->queue_capacity == 0)
    return FORMOSA_WG_ERR_BAD_EDGE_DESC;

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
    return FORMOSA_WG_ERR_DST_NODE_NOT_FOUND;
  if (dst_index >= pool->node_queue_count)
    return FORMOSA_WG_ERR_DST_QUEUE_OUT_OF_RANGE;

  __global uchar *queue_base =
      (__global uchar *)(uintptr_t)pool->node_queue_addr;
  __global struct NodeInputQueue *queue =
      (__global struct NodeInputQueue *)(queue_base +
                                         ((size_t)dst_index *
                                          (size_t)pool->node_queue_stride));

  uint record_size = queue->record_size;
  if (queue->capacity == 0 || record_size == 0 || queue->records_addr == 0) {
    return FORMOSA_WG_ERR_BAD_QUEUE;
  }
  if (record_size != edge->record_size)
    return FORMOSA_WG_ERR_RECORD_SIZE_MISMATCH;
  if (queue->ready_sequence_addr == 0)
    return FORMOSA_WG_ERR_BAD_READY_SEQUENCE;

  /* WorkGraph ring queue protocol:
     1. Reserve a monotonically increasing logical ticket.
     2. Write the payload into the ticket's physical ring slot.
     3. Publish ticket + 1 in the slot's ready sequence.
     Firmware MaxRecords admission and active output reservations guarantee
     that this producer has downstream capacity before launch; the device helper
     intentionally does not re-check consumed_head, which is firmware-updated
     metadata and may be stale in an SM cache.
     Firmware advances ready_tail over the contiguous published prefix after it
     has observed ready_sequence entries and made the corresponding payload
     range visible to later consumer kernels. */
  volatile __global uint *reserve_tail =
      (volatile __global uint *)&queue->reserve_tail;
  uint ticket = formosa_wg_reserve_ticket(reserve_tail);
  uint slot = ticket % queue->capacity;
  uint record_stride =
      queue->record_stride != 0 ? queue->record_stride : record_size;
  if (record_stride < record_size)
    return FORMOSA_WG_ERR_RECORD_SIZE_MISMATCH;
  __global uchar *dst = (__global uchar *)(uintptr_t)queue->records_addr +
                        ((size_t)slot * record_stride);
  const __private uchar *src = (const __private uchar *)record;
  if ((((uintptr_t)dst | (uintptr_t)src | record_size) & 3u) == 0) {
    volatile __global uint *dst_words = (volatile __global uint *)dst;
    const __private uint *src_words = (const __private uint *)src;
    for (uint i = 0; i < record_size / sizeof(uint); i++) {
      dst_words[i] = src_words[i];
    }
  } else {
    volatile __global uchar *volatile_dst = (volatile __global uchar *)dst;
    for (uint i = 0; i < record_size; i++) {
      volatile_dst[i] = src[i];
    }
  }

  volatile __global uint *ready_sequence =
      (volatile __global uint *)(uintptr_t)queue->ready_sequence_addr;
  /* Publish after the payload slot has been written. ready_sequence storage is
     non-cacheable so firmware can observe producer progress while the dispatch
     is still active. WorkGraph record payload rings are also non-cacheable, so
     firmware can advance ready_tail without flushing payload ranges. */
  formosa_wg_publish_ready(&ready_sequence[slot], ticket + 1);
  return FORMOSA_WG_SUCCESS;
}
