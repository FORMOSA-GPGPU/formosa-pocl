#ifndef FORMOSA_WORK_GRAPH_ABI_H_
#define FORMOSA_WORK_GRAPH_ABI_H_

#include "../wg_info.h"

/* Keep this ABI mirror in sync with formosa-hal/graph.h. */
struct WorkGraphNodeContext {
  uint32_t node_id;
  uint32_t launch_mode;

  uint32_t input_record_count;
  uint32_t input_record_size;

  uint64_t input_records;

  uint64_t graph_runtime;
  uint64_t edge_table;

  uint32_t edge_count;
  uint32_t dispatch_id;

  uint32_t input_record_stride;
  uint32_t input_record_head;
  uint32_t work_items_per_input_record;
  uint32_t reserved1;
};

_Static_assert(sizeof(struct WorkGraphNodeContext) == 64,
               "WorkGraphNodeContext size mismatch");
_Static_assert(offsetof(struct WorkGraphNodeContext, input_records) == 16,
               "WorkGraphNodeContext.input_records offset mismatch");
_Static_assert(offsetof(struct WorkGraphNodeContext, graph_runtime) == 24,
               "WorkGraphNodeContext.graph_runtime offset mismatch");
_Static_assert(offsetof(struct WorkGraphNodeContext, edge_table) == 32,
               "WorkGraphNodeContext.edge_table offset mismatch");
_Static_assert(offsetof(struct WorkGraphNodeContext, input_record_stride) == 48,
               "WorkGraphNodeContext.input_record_stride offset mismatch");
_Static_assert(offsetof(struct WorkGraphNodeContext, input_record_head) == 52,
               "WorkGraphNodeContext.input_record_head offset mismatch");
_Static_assert(
    offsetof(struct WorkGraphNodeContext, work_items_per_input_record) == 56,
    "WorkGraphNodeContext.work_items_per_input_record offset mismatch");

struct WorkGraphWGInfo {
  struct WGInfo base;
  uint64_t work_graph_ctx;
};

_Static_assert(offsetof(struct WorkGraphWGInfo, work_graph_ctx) == 0x80,
               "WorkGraphWGInfo.work_graph_ctx offset mismatch");

static inline struct WorkGraphWGInfo *get_work_graph_wg_info() {
  return (struct WorkGraphWGInfo *)get_wg_info();
}

#endif
