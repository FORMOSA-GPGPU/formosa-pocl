#ifndef CL_FSA_WORK_GRAPH_H
#define CL_FSA_WORK_GRAPH_H

#include <CL/cl.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Extension name.
 */
#define CL_FSA_WORK_GRAPH_EXTENSION_NAME "cl_fsa_work_graph"

/*
 * Opaque host-side object types.
 *
 * v1 ownership model:
 *   - graph owns all nodes and edges created under it
 *   - v1 does not expose separate node/edge release APIs
 *   - only clfsaReleaseGraph releases the graph object
 */
typedef struct _cl_fsa_graph *cl_fsa_graph;
typedef struct _cl_fsa_graph_node *cl_fsa_graph_node;
typedef struct _cl_fsa_graph_edge *cl_fsa_graph_edge;

/*
 * Graph flags.
 */
typedef cl_bitfield cl_fsa_graph_flags;

#define CL_FSA_GRAPH_STATIC_TOPOLOGY   (1ull << 0)
#define CL_FSA_GRAPH_ENABLE_PROFILING  (1ull << 1)

/*
 * Graph properties.
 */
typedef struct _cl_fsa_graph_properties {
    cl_fsa_graph_flags flags;
    cl_uint max_nodes;
    cl_uint max_edges;
    cl_uint max_root_records;
} cl_fsa_graph_properties;

/*
 * Node flags.
 */
typedef cl_bitfield cl_fsa_node_flags;

#define CL_FSA_NODE_ROOT_CAPABLE  (1ull << 0)
#define CL_FSA_NODE_ALLOW_EMIT    (1ull << 1)

/*
 * Node launch modes.
 */
typedef cl_uint cl_fsa_node_launch_mode;

#define CL_FSA_NODE_LAUNCH_THREAD        0
#define CL_FSA_NODE_LAUNCH_COALESCING    1
#define CL_FSA_NODE_LAUNCH_BROADCASTING  2

/*
 * Node properties.
 */
typedef struct _cl_fsa_node_properties {
    cl_fsa_node_flags flags;
    size_t input_record_size;
    cl_fsa_node_launch_mode launch_mode;
    cl_uint max_input_records_per_dispatch;
    cl_uint max_active_dispatches;
} cl_fsa_node_properties;

/*
 * Edge flags.
 */
typedef cl_bitfield cl_fsa_edge_flags;

#define CL_FSA_EDGE_FIFO (1ull << 0)

/*
 * Edge properties.
 */
typedef struct _cl_fsa_edge_properties {
    cl_fsa_edge_flags flags;
    size_t record_size;
    cl_uint queue_capacity;
    cl_uint max_records_per_src_dispatch;
} cl_fsa_edge_properties;

/*
 * Root input descriptor.
 */
typedef struct _cl_fsa_root_input {
    cl_uint target_node_id;
    cl_uint record_count;
    cl_mem records;
    size_t records_offset;
    size_t record_stride;
} cl_fsa_root_input;

/*
 * clGetGraphInfo FSA param_name values.
 */
#define CL_FSA_GRAPH_INFO_FLAGS              0x5000
#define CL_FSA_GRAPH_INFO_NUM_NODES          0x5001
#define CL_FSA_GRAPH_INFO_NUM_EDGES          0x5002
#define CL_FSA_GRAPH_INFO_MAX_NODES          0x5003
#define CL_FSA_GRAPH_INFO_MAX_EDGES          0x5004
#define CL_FSA_GRAPH_INFO_MAX_ROOT_RECORDS   0x5005
#define CL_FSA_GRAPH_INFO_STATUS             0x5006
#define CL_FSA_GRAPH_INFO_LAST_ERROR         0x5007

/*
 * Command type for graph launch.
 */
#define CL_COMMAND_FSA_GRAPH_LAUNCH          0x120F

/*
 * Graph status values.
 */
#define CL_FSA_GRAPH_STATUS_IDLE             0
#define CL_FSA_GRAPH_STATUS_RUNNING          1
#define CL_FSA_GRAPH_STATUS_COMPLETED        2
#define CL_FSA_GRAPH_STATUS_ERROR            3

/*
 * API Declarations.
 */

extern CL_API_ENTRY cl_fsa_graph CL_API_CALL
clfsaCreateGraph(
    cl_context context,
    cl_device_id device,
    const cl_fsa_graph_properties *properties,
    cl_int *errcode_ret);

extern CL_API_ENTRY cl_fsa_graph_node CL_API_CALL
clfsaCreateGraphKernelNode(
    cl_fsa_graph graph,
    cl_kernel kernel,
    cl_uint node_id,
    cl_uint work_dim,
    const size_t *global_work_offset,
    const size_t *global_work_size,
    const size_t *local_work_size,
    const cl_fsa_node_properties *properties,
    cl_int *errcode_ret);


extern CL_API_ENTRY cl_fsa_graph_edge CL_API_CALL
clfsaCreateGraphEdge(
    cl_fsa_graph graph,
    cl_fsa_graph_node src_node,
    cl_fsa_graph_node dst_node,
    cl_uint edge_id,
    const cl_fsa_edge_properties *properties,
    cl_int *errcode_ret);

extern CL_API_ENTRY cl_int CL_API_CALL
clfsaEnqueueGraphLaunch(
    cl_command_queue command_queue,
    cl_fsa_graph graph,
    cl_uint num_root_inputs,
    const cl_fsa_root_input *root_inputs,
    cl_uint num_events_in_wait_list,
    const cl_event *event_wait_list,
    cl_event *event);

extern CL_API_ENTRY cl_int CL_API_CALL
clfsaGetGraphInfo(
    cl_fsa_graph graph,
    cl_uint param_name,
    size_t param_value_size,
    void *param_value,
    size_t *param_value_size_ret);

extern CL_API_ENTRY cl_int CL_API_CALL
clfsaRetainGraph(
    cl_fsa_graph graph);

extern CL_API_ENTRY cl_int CL_API_CALL
clfsaReleaseGraph(
    cl_fsa_graph graph);

/*
 * API Typedefs.
 */

typedef cl_fsa_graph (CL_API_CALL *clfsaCreateGraph_fn)(
    cl_context context,
    cl_device_id device,
    const cl_fsa_graph_properties *properties,
    cl_int *errcode_ret);

typedef cl_fsa_graph_node (CL_API_CALL *clfsaCreateGraphKernelNode_fn)(
    cl_fsa_graph graph,
    cl_kernel kernel,
    cl_uint node_id,
    cl_uint work_dim,
    const size_t *global_work_offset,
    const size_t *global_work_size,
    const size_t *local_work_size,
    const cl_fsa_node_properties *properties,
    cl_int *errcode_ret);

typedef cl_fsa_graph_edge (CL_API_CALL *clfsaCreateGraphEdge_fn)(
    cl_fsa_graph graph,
    cl_fsa_graph_node src_node,
    cl_fsa_graph_node dst_node,
    cl_uint edge_id,
    const cl_fsa_edge_properties *properties,
    cl_int *errcode_ret);

typedef cl_int (CL_API_CALL *clfsaEnqueueGraphLaunch_fn)(
    cl_command_queue command_queue,
    cl_fsa_graph graph,
    cl_uint num_root_inputs,
    const cl_fsa_root_input *root_inputs,
    cl_uint num_events_in_wait_list,
    const cl_event *event_wait_list,
    cl_event *event);

typedef cl_int (CL_API_CALL *clfsaGetGraphInfo_fn)(
    cl_fsa_graph graph,
    cl_uint param_name,
    size_t param_value_size,
    void *param_value,
    size_t *param_value_size_ret);

typedef cl_int (CL_API_CALL *clfsaRetainGraph_fn)(
    cl_fsa_graph graph);

typedef cl_int (CL_API_CALL *clfsaReleaseGraph_fn)(
    cl_fsa_graph graph);

#ifdef __cplusplus
}
#endif

#endif /* CL_FSA_WORK_GRAPH_H */
