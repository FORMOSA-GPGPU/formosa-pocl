#ifndef CL_FORMOSA_WORK_GRAPH_H
#define CL_FORMOSA_WORK_GRAPH_H

#include <CL/cl.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Extension name.
 */
#define CL_FORMOSA_WORK_GRAPH_EXTENSION_NAME "cl_formosa_work_graph"

/*
 * Opaque host-side object types.
 *
 * v1 ownership model:
 *   - graph owns all nodes and edges created under it
 *   - v1 does not expose separate node/edge release APIs
 *   - only clReleaseWorkGraphFORMOSA releases the graph object
 */
typedef struct _cl_work_graph_formosa *cl_work_graph_formosa;
typedef struct _cl_work_graph_node_formosa *cl_work_graph_node_formosa;
typedef struct _cl_work_graph_edge_formosa *cl_work_graph_edge_formosa;

/*
 * Graph flags.
 */
typedef cl_bitfield cl_work_graph_flags_formosa;

#define CL_GRAPH_STATIC_TOPOLOGY_FORMOSA (1ull << 0)
#define CL_GRAPH_ENABLE_PROFILING_FORMOSA (1ull << 1)

/*
 * Graph properties.
 */
typedef struct _cl_work_graph_properties_formosa {
  cl_work_graph_flags_formosa flags;
  cl_uint max_nodes;
  cl_uint max_edges;
  cl_uint max_root_records;
} cl_work_graph_properties_formosa;

/*
 * Node flags.
 */
typedef cl_bitfield cl_work_graph_node_flags_formosa;

#define CL_NODE_ROOT_CAPABLE_FORMOSA (1ull << 0)
#define CL_NODE_ALLOW_EMIT_FORMOSA (1ull << 1)
#define CL_NODE_GLOBAL_CACHE_BEFORE_FORMOSA (1ull << 2)

/*
 * Node launch modes.
 */
typedef cl_uint cl_work_graph_node_launch_mode_formosa;

#define CL_NODE_LAUNCH_THREAD_FORMOSA 0
/* COALESCING launches one fixed-size workgroup over a batch of input records.
 * The node's max_input_records_per_dispatch is the maximum batch size.
 */
#define CL_NODE_LAUNCH_COALESCING_FORMOSA 1
#define CL_NODE_LAUNCH_BROADCASTING_FORMOSA 2

#define CL_FORMOSA_WORK_GRAPH_HAS_WORK_ITEMS_PER_INPUT_RECORD 1

/*
 * Node properties.
 *
 * max_input_records_per_dispatch caps how many input records one dispatch may
 * consume. For THREAD and BROADCASTING nodes, 0 means adaptive runtime
 * batching: firmware chooses a dispatch size from queue readiness, ring
 * contiguity, WGI availability, packet limits, and output admission. For
 * COALESCING nodes, this value must be nonzero and is the maximum number of
 * records visible to one workgroup dispatch.
 *
 * work_items_per_input_record controls broadcasting-style launch expansion. A
 * value of 0 means the default of 1. For broadcasting nodes, each input record
 * in a dispatch is mapped onto K logical x-dimension work-items. The local
 * size determines how those work-items are grouped into physical workgroups.
 * Device code can derive the current record index and per-record work-item id
 * from the WorkGraph helper APIs.
 */
typedef struct _cl_work_graph_node_properties_formosa {
  cl_work_graph_node_flags_formosa flags;
  size_t input_record_size;
  cl_work_graph_node_launch_mode_formosa launch_mode;
  cl_uint max_input_records_per_dispatch;
  cl_uint max_active_dispatches;
  cl_uint work_items_per_input_record;
} cl_work_graph_node_properties_formosa;

/*
 * Edge flags.
 */
typedef cl_bitfield cl_work_graph_edge_flags_formosa;

#define CL_EDGE_FIFO_FORMOSA (1ull << 0)

/*
 * Edge properties.
 */
typedef struct _cl_work_graph_edge_properties_formosa {
  cl_work_graph_edge_flags_formosa flags;
  size_t record_size;
  /* Destination queue capacity hint, in records. This is storage capacity, not
   * a dispatch batch size. A value of 0 asks PoCL to infer a conservative
   * capacity from the source queue capacity and max_records_per_src_record.
   */
  cl_uint queue_capacity;
  /* Maximum number of records emitted through this edge per source input
   * record. Firmware scales this by the actual source dispatch record count
   * for admission. A value of 0 defaults to 1.
   */
  cl_uint max_records_per_src_record;
} cl_work_graph_edge_properties_formosa;

/*
 * Root input descriptor.
 */
typedef struct _cl_work_graph_root_input_formosa {
  cl_uint target_node_id;
  cl_uint record_count;
  cl_mem records;
  size_t records_offset;
  size_t record_stride;
} cl_work_graph_root_input_formosa;

/*
 * clGetGraphInfo FSA param_name values.
 */
#define CL_GRAPH_INFO_FLAGS_FORMOSA 0x5000
#define CL_GRAPH_INFO_NUM_NODES_FORMOSA 0x5001
#define CL_GRAPH_INFO_NUM_EDGES_FORMOSA 0x5002
#define CL_GRAPH_INFO_MAX_NODES_FORMOSA 0x5003
#define CL_GRAPH_INFO_MAX_EDGES_FORMOSA 0x5004
#define CL_GRAPH_INFO_MAX_ROOT_RECORDS_FORMOSA 0x5005
#define CL_GRAPH_INFO_STATUS_FORMOSA 0x5006
#define CL_GRAPH_INFO_LAST_ERROR_FORMOSA 0x5007
#define CL_GRAPH_INFO_MAX_ACTIVE_DISPATCHES_FORMOSA 0x5008
#define CL_GRAPH_INFO_MAX_NODE_ACTIVE_DISPATCHES_FORMOSA 0x5009

/*
 * Command type for graph launch.
 */
#define CL_COMMAND_GRAPH_LAUNCH_FORMOSA 0x120F

/*
 * Graph status values.
 */
#define CL_GRAPH_STATUS_IDLE_FORMOSA 0
#define CL_GRAPH_STATUS_RUNNING_FORMOSA 1
#define CL_GRAPH_STATUS_COMPLETED_FORMOSA 2
#define CL_GRAPH_STATUS_ERROR_FORMOSA 3

/*
 * API Declarations.
 */

extern CL_API_ENTRY cl_work_graph_formosa CL_API_CALL clCreateWorkGraphFORMOSA(
    cl_context context, cl_device_id device,
    const cl_work_graph_properties_formosa *properties, cl_int *errcode_ret);

extern CL_API_ENTRY cl_work_graph_node_formosa CL_API_CALL
clCreateWorkGraphKernelNodeFORMOSA(
    cl_work_graph_formosa graph, cl_kernel kernel, cl_uint node_id,
    cl_uint work_dim, const size_t *global_work_offset,
    const size_t *global_work_size, const size_t *local_work_size,
    const cl_work_graph_node_properties_formosa *properties,
    cl_int *errcode_ret);

extern CL_API_ENTRY cl_work_graph_edge_formosa CL_API_CALL
clCreateWorkGraphEdgeFORMOSA(
    cl_work_graph_formosa graph, cl_work_graph_node_formosa src_node,
    cl_work_graph_node_formosa dst_node, cl_uint edge_id,
    const cl_work_graph_edge_properties_formosa *properties,
    cl_int *errcode_ret);

extern CL_API_ENTRY cl_int CL_API_CALL clEnqueueWorkGraphLaunchFORMOSA(
    cl_command_queue command_queue, cl_work_graph_formosa graph,
    cl_uint num_root_inputs,
    const cl_work_graph_root_input_formosa *root_inputs,
    cl_uint num_events_in_wait_list, const cl_event *event_wait_list,
    cl_event *event);

extern CL_API_ENTRY cl_int CL_API_CALL clGetWorkGraphInfoFORMOSA(
    cl_work_graph_formosa graph, cl_uint param_name, size_t param_value_size,
    void *param_value, size_t *param_value_size_ret);

extern CL_API_ENTRY cl_int CL_API_CALL
clRetainWorkGraphFORMOSA(cl_work_graph_formosa graph);

extern CL_API_ENTRY cl_int CL_API_CALL
clReleaseWorkGraphFORMOSA(cl_work_graph_formosa graph);

/*
 * API Typedefs.
 */

typedef cl_work_graph_formosa(CL_API_CALL *clCreateWorkGraphFORMOSA_fn)(
    cl_context context, cl_device_id device,
    const cl_work_graph_properties_formosa *properties, cl_int *errcode_ret);

typedef cl_work_graph_node_formosa(
    CL_API_CALL *clCreateWorkGraphKernelNodeFORMOSA_fn)(
    cl_work_graph_formosa graph, cl_kernel kernel, cl_uint node_id,
    cl_uint work_dim, const size_t *global_work_offset,
    const size_t *global_work_size, const size_t *local_work_size,
    const cl_work_graph_node_properties_formosa *properties,
    cl_int *errcode_ret);

typedef cl_work_graph_edge_formosa(
    CL_API_CALL *clCreateWorkGraphEdgeFORMOSA_fn)(
    cl_work_graph_formosa graph, cl_work_graph_node_formosa src_node,
    cl_work_graph_node_formosa dst_node, cl_uint edge_id,
    const cl_work_graph_edge_properties_formosa *properties,
    cl_int *errcode_ret);

typedef cl_int(CL_API_CALL *clEnqueueWorkGraphLaunchFORMOSA_fn)(
    cl_command_queue command_queue, cl_work_graph_formosa graph,
    cl_uint num_root_inputs,
    const cl_work_graph_root_input_formosa *root_inputs,
    cl_uint num_events_in_wait_list, const cl_event *event_wait_list,
    cl_event *event);

typedef cl_int(CL_API_CALL *clGetWorkGraphInfoFORMOSA_fn)(
    cl_work_graph_formosa graph, cl_uint param_name, size_t param_value_size,
    void *param_value, size_t *param_value_size_ret);

typedef cl_int(CL_API_CALL *clRetainWorkGraphFORMOSA_fn)(
    cl_work_graph_formosa graph);

typedef cl_int(CL_API_CALL *clReleaseWorkGraphFORMOSA_fn)(
    cl_work_graph_formosa graph);

#ifdef __cplusplus
}
#endif

#endif /* CL_FORMOSA_WORK_GRAPH_H */
