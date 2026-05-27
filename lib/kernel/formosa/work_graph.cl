#include "wg_info.h"

uint formosa_get_record_count(void) {
  struct WGInfo *info = get_wg_info();
  struct WorkGraphNodeContext *ctx = (struct WorkGraphNodeContext *)info->work_graph_ctx;
  if (ctx == NULL) return 0;
  return ctx->input_record_count;
}

int formosa_get_record(uint index, void *record_out, size_t record_size) {
  struct WGInfo *info = get_wg_info();
  struct WorkGraphNodeContext *ctx = (struct WorkGraphNodeContext *)info->work_graph_ctx;
  
  if (ctx == NULL) return -1;
  if (index >= ctx->input_record_count) return -1;
  if (record_size != (size_t)ctx->input_record_size) return -1;

  uchar *src = (uchar *)ctx->input_records + (index * record_size);
  uchar *dst = (uchar *)record_out;

  for (size_t i = 0; i < record_size; i++) {
    dst[i] = src[i];
  }

  return 0;
}

int formosa_emit(uint edge_id, const void *record) {
  // Placeholder for Phase 2
  return -1;
}
