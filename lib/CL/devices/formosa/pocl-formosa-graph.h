#ifndef POCL_FORMOSA_GRAPH_H
#define POCL_FORMOSA_GRAPH_H

#include "pocl_cl.h"

cl_int pocl_formosa_fsa_create_graph(cl_fsa_graph graph, const cl_fsa_graph_properties *properties);
cl_int pocl_formosa_fsa_create_node(cl_fsa_graph_node node, cl_kernel kernel, cl_uint node_id, cl_uint work_dim, const size_t *global_offset, const size_t *global_size, const size_t *local_size, const cl_fsa_node_properties *properties);
cl_int pocl_formosa_fsa_create_edge(cl_fsa_graph_edge edge, cl_fsa_graph_node src, cl_fsa_graph_node dst, cl_uint edge_id, const cl_fsa_edge_properties *properties);
cl_int pocl_formosa_fsa_get_info(cl_fsa_graph graph, cl_uint param, size_t size, void *value, size_t *size_ret);
cl_int pocl_formosa_fsa_free_graph(cl_fsa_graph graph);

void pocl_formosa_fsa_run_graph(void *data, _cl_command_node *cmd);

#endif /* POCL_FORMOSA_GRAPH_H */
