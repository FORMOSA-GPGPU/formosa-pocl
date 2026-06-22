#ifndef CL_FORMOSA_WORK_GRAPH_DEVICE_H
#define CL_FORMOSA_WORK_GRAPH_DEVICE_H

/*
 * Device-side Helper APIs for Formosa Work Graph.
 * These are intended to be used from OpenCL C kernels.
 */

/* Returns the number of records in the current node dispatch. For COALESCING
 * nodes, this is the number of input records visible to the current workgroup.
 */
uint formosa_get_record_count(void);

/* Returns the broadcasting work-item expansion for the current node dispatch.
 */
uint formosa_get_record_work_item_count(void);

/* Returns the input record index for this work-item. */
uint formosa_get_current_record_index(void);

/* Returns this work-item's id within its expanded input record. */
uint formosa_get_current_record_work_item_id(void);

/*
 * Copies the record at 'index' into 'record_out'. COALESCING kernels may use
 * this to iterate over the dispatch's record array.
 * Returns 0 on success, non-zero on error.
 */
int formosa_get_record(uint index, __private void *record_out,
                       size_t record_size);

/*
 * Emits a record to the specified edge.
 * Returns 0 on success, non-zero on error.
 */
int formosa_emit(uint edge_id, const __private void *record);

#endif /* CL_FORMOSA_WORK_GRAPH_DEVICE_H */
