#ifndef FORMOSA_WG_INFO_H_
#define FORMOSA_WG_INFO_H_

#include <stddef.h>
#include <stdint.h>

/* Keep this ABI mirror in sync with formosa-hal/formosa-graph.h. */
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

  uint32_t graph_launch_id;
  uint32_t input_record_head;
  uint32_t workgroups_per_input_record;
  uint32_t reserved0;
};

_Static_assert(sizeof(struct WorkGraphNodeContext) == 64,
               "WorkGraphNodeContext size mismatch");
_Static_assert(offsetof(struct WorkGraphNodeContext, input_records) == 16,
               "WorkGraphNodeContext.input_records offset mismatch");
_Static_assert(offsetof(struct WorkGraphNodeContext, graph_runtime) == 24,
               "WorkGraphNodeContext.graph_runtime offset mismatch");
_Static_assert(offsetof(struct WorkGraphNodeContext, edge_table) == 32,
               "WorkGraphNodeContext.edge_table offset mismatch");
_Static_assert(offsetof(struct WorkGraphNodeContext, graph_launch_id) == 48,
               "WorkGraphNodeContext.graph_launch_id offset mismatch");
_Static_assert(offsetof(struct WorkGraphNodeContext, input_record_head) == 52,
               "WorkGraphNodeContext.input_record_head offset mismatch");
_Static_assert(
    offsetof(struct WorkGraphNodeContext, workgroups_per_input_record) == 56,
    "WorkGraphNodeContext.workgroups_per_input_record offset mismatch");

struct WGInfo {
  uint32_t dim;
  uint32_t wg_id[3];
  uint32_t local_size[3];
  uint32_t num_groups[3];
  uint32_t global_offset[3];
  uint32_t num_threads;
  void *trampoline;
  void *kargs;
  uint64_t stack_base; // stack base address for this work-group
  uint32_t stack_size; // stack size for a work-item
  uint64_t local_memory_base;
  uint32_t local_memory_size;
  char *printf_buffer;
  uint32_t *printf_buffer_position;
  uint32_t printf_buffer_capacity;
  uint32_t reserved0;
  uint64_t work_graph_ctx;
};

_Static_assert(offsetof(struct WGInfo, stack_base) == 0x48,
               "WGInfo.stack_base offset mismatch");
_Static_assert(offsetof(struct WGInfo, printf_buffer) == 0x68,
               "WGInfo.printf_buffer offset mismatch");
_Static_assert(offsetof(struct WGInfo, work_graph_ctx) == 0x80,
               "WGInfo.work_graph_ctx offset mismatch");

static inline struct WGInfo *get_wg_info() {
  struct WGInfo *info;
  __asm__ volatile("csrr %0, mscratch" : "=r"(info));
  return info;
}

#endif
