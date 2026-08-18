#ifndef POCL_WORKGRAPH_PRIVATE_H
#define POCL_WORKGRAPH_PRIVATE_H

#include "CL/cl_formosa_work_graph.h"
#include "pocl_cl.h"

struct pocl_work_graph_backend_ops
{
  cl_int (*create_graph) (
    cl_work_graph_formosa graph,
    const cl_work_graph_properties_formosa *properties);
  cl_int (*create_node) (
    cl_work_graph_node_formosa node, cl_kernel kernel, cl_uint node_id,
    cl_uint work_dim, const size_t *global_offset, const size_t *global_size,
    const size_t *local_size,
    const cl_work_graph_node_properties_formosa *properties);
  cl_int (*create_edge) (
    cl_work_graph_edge_formosa edge, cl_work_graph_node_formosa src,
    cl_work_graph_node_formosa dst, cl_uint edge_id,
    const cl_work_graph_edge_properties_formosa *properties);
  cl_int (*get_info) (cl_work_graph_formosa graph, cl_uint param, size_t size,
                      void *value, size_t *size_ret);
  cl_int (*free_graph) (cl_work_graph_formosa graph);
  cl_int (*run) (void *device_data, _cl_command_node *command);
};

struct _cl_work_graph_formosa
{
  POCL_ICD_OBJECT
  POCL_OBJECT;
  cl_context context;
  cl_device_id device;
  const struct pocl_work_graph_backend_ops *ops;
  void *backend_data;
  struct _cl_work_graph_node_formosa *nodes;
  struct _cl_work_graph_edge_formosa *edges;
};

struct _cl_work_graph_node_formosa
{
  POCL_ICD_OBJECT
  cl_work_graph_formosa graph;
  cl_kernel kernel;
  cl_uint node_id;
  cl_uint work_dim;
  size_t global_work_offset[3];
  size_t global_work_size[3];
  size_t local_work_size[3];
  cl_work_graph_node_properties_formosa properties;
  void *backend_data;
  struct _cl_work_graph_node_formosa *next;
};

struct _cl_work_graph_edge_formosa
{
  POCL_ICD_OBJECT
  cl_work_graph_formosa graph;
  cl_work_graph_node_formosa src_node;
  cl_work_graph_node_formosa dst_node;
  cl_uint edge_id;
  cl_work_graph_edge_properties_formosa properties;
  void *backend_data;
  struct _cl_work_graph_edge_formosa *next;
};

struct pocl_work_graph_launch
{
  cl_work_graph_formosa graph;
  cl_uint num_root_inputs;
  cl_work_graph_root_input_formosa *root_inputs;
};

POCL_EXPORT cl_int
pocl_formosa_release_work_graph (cl_work_graph_formosa graph);

void *pocl_work_graph_get_extension_function_address (const char *func_name);

#endif
