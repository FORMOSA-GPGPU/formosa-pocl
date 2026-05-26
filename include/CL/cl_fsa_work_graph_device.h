#ifndef CL_FSA_WORK_GRAPH_DEVICE_H
#define CL_FSA_WORK_GRAPH_DEVICE_H

/*
 * Device-side Helper APIs for Formosa Work Graph.
 * These are intended to be used from OpenCL C kernels.
 */

/* Returns the number of records in the current node dispatch. */
uint clfsa_get_record_count(void);

/* 
 * Copies the record at 'index' into 'record_out'.
 * Returns 0 on success, non-zero on error.
 */
int clfsa_get_record(uint index, void *record_out, size_t record_size);

/*
 * Emits a record to the specified edge.
 * Returns 0 on success, non-zero on error.
 */
int clfsa_emit(uint edge_id, const void *record);

#endif /* CL_FSA_WORK_GRAPH_DEVICE_H */
