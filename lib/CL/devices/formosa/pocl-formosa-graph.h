#ifndef POCL_FORMOSA_GRAPH_H
#define POCL_FORMOSA_GRAPH_H

#include "pocl_cl.h"

cl_int pocl_formosa_create_work_graph(
    cl_work_graph_formosa graph,
    const cl_work_graph_properties_formosa *properties);
cl_int pocl_formosa_create_work_graph_node(
    cl_work_graph_node_formosa node, cl_kernel kernel, cl_uint node_id,
    cl_uint work_dim, const size_t *global_offset, const size_t *global_size,
    const size_t *local_size,
    const cl_work_graph_node_properties_formosa *properties);
cl_int pocl_formosa_create_work_graph_edge(
    cl_work_graph_edge_formosa edge, cl_work_graph_node_formosa src,
    cl_work_graph_node_formosa dst, cl_uint edge_id,
    const cl_work_graph_edge_properties_formosa *properties);
cl_int pocl_formosa_get_work_graph_info(cl_work_graph_formosa graph,
                                        cl_uint param, size_t size, void *value,
                                        size_t *size_ret);
cl_int pocl_formosa_free_work_graph(cl_work_graph_formosa graph);

cl_int pocl_formosa_run_work_graph(void *data, _cl_command_node *cmd);

#endif /* POCL_FORMOSA_GRAPH_H */
