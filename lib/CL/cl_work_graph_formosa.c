#include "CL/cl_formosa_work_graph.h"
#include "pocl_cl.h"
#include "pocl_mem_management.h"
#include "pocl_shared.h"
#include "pocl_util.h"
#include <string.h>

CL_API_ENTRY cl_work_graph_formosa CL_API_CALL POname(clCreateWorkGraphFORMOSA)(
    cl_context context, cl_device_id device,
    const cl_work_graph_properties_formosa *properties, cl_int *errcode_ret) {
  int errcode = CL_SUCCESS;
  cl_work_graph_formosa graph = NULL;

  POCL_GOTO_ERROR_COND((!IS_CL_OBJECT_VALID(context)), CL_INVALID_CONTEXT);
  POCL_GOTO_ERROR_COND((device == NULL), CL_INVALID_DEVICE);

  if (device->ops->work_graph_formosa_ops == NULL ||
      device->ops->work_graph_formosa_ops->create_graph == NULL) {
    errcode = CL_INVALID_OPERATION;
    goto ERROR;
  }

  graph =
      (cl_work_graph_formosa)calloc(1, sizeof(struct _cl_work_graph_formosa));
  if (graph == NULL) {
    errcode = CL_OUT_OF_HOST_MEMORY;
    goto ERROR;
  }

  POCL_INIT_OBJECT(graph, context);
  graph->context = context;
  graph->device = device;
  graph->ops = device->ops->work_graph_formosa_ops;

  POname(clRetainContext)(context);

  errcode = graph->ops->create_graph(graph, properties);
  if (errcode != CL_SUCCESS) {
    goto ERROR;
  }

  if (errcode_ret)
    *errcode_ret = CL_SUCCESS;
  return graph;

ERROR:
  if (graph) {
    POname(clReleaseContext)(context);
    free(graph);
  }
  if (errcode_ret)
    *errcode_ret = errcode;
  return NULL;
}
POsym(clCreateWorkGraphFORMOSA)

    CL_API_ENTRY cl_work_graph_node_formosa CL_API_CALL
    POname(clCreateWorkGraphKernelNodeFORMOSA)(
        cl_work_graph_formosa graph, cl_kernel kernel, cl_uint node_id,
        cl_uint work_dim, const size_t *global_work_offset,
        const size_t *global_work_size, const size_t *local_work_size,
        const cl_work_graph_node_properties_formosa *properties,
        cl_int *errcode_ret) {
  int errcode = CL_SUCCESS;
  cl_work_graph_node_formosa node = NULL;

  POCL_GOTO_ERROR_COND((graph == NULL), CL_INVALID_VALUE);
  POCL_GOTO_ERROR_COND((!IS_CL_OBJECT_VALID(kernel)), CL_INVALID_KERNEL);

  if (graph->ops->create_node == NULL) {
    errcode = CL_INVALID_OPERATION;
    goto ERROR;
  }

  node = (cl_work_graph_node_formosa)calloc(
      1, sizeof(struct _cl_work_graph_node_formosa));
  if (node == NULL) {
    errcode = CL_OUT_OF_HOST_MEMORY;
    goto ERROR;
  }

  POCL_INIT_ICD_OBJECT(node, graph);
  node->graph = graph;
  node->kernel = kernel;
  node->node_id = node_id;
  node->work_dim = work_dim;
  if (global_work_offset)
    memcpy(node->global_work_offset, global_work_offset,
           sizeof(size_t) * work_dim);
  if (global_work_size)
    memcpy(node->global_work_size, global_work_size, sizeof(size_t) * work_dim);
  if (local_work_size)
    memcpy(node->local_work_size, local_work_size, sizeof(size_t) * work_dim);
  if (properties)
    node->properties = *properties;

  POname(clRetainKernel)(kernel);

  errcode = graph->ops->create_node(node, kernel, node_id, work_dim,
                                    global_work_offset, global_work_size,
                                    local_work_size, properties);
  if (errcode != CL_SUCCESS) {
    goto ERROR;
  }

  /* Link to graph's node list */
  node->next = graph->nodes;
  graph->nodes = node;

  if (errcode_ret)
    *errcode_ret = CL_SUCCESS;
  return node;

ERROR:
  if (node) {
    POname(clReleaseKernel)(kernel);
    free(node);
  }
  if (errcode_ret)
    *errcode_ret = errcode;
  return NULL;
}
POsym(clCreateWorkGraphKernelNodeFORMOSA)

    CL_API_ENTRY cl_work_graph_edge_formosa CL_API_CALL
    POname(clCreateWorkGraphEdgeFORMOSA)(
        cl_work_graph_formosa graph, cl_work_graph_node_formosa src_node,
        cl_work_graph_node_formosa dst_node, cl_uint edge_id,
        const cl_work_graph_edge_properties_formosa *properties,
        cl_int *errcode_ret) {
  int errcode = CL_SUCCESS;
  cl_work_graph_edge_formosa edge = NULL;

  POCL_GOTO_ERROR_COND((graph == NULL), CL_INVALID_VALUE);
  POCL_GOTO_ERROR_COND((src_node == NULL || dst_node == NULL),
                       CL_INVALID_VALUE);

  if (graph->ops->create_edge == NULL) {
    errcode = CL_INVALID_OPERATION;
    goto ERROR;
  }

  edge = (cl_work_graph_edge_formosa)calloc(
      1, sizeof(struct _cl_work_graph_edge_formosa));
  if (edge == NULL) {
    errcode = CL_OUT_OF_HOST_MEMORY;
    goto ERROR;
  }

  POCL_INIT_ICD_OBJECT(edge, graph);
  edge->graph = graph;
  edge->src_node = src_node;
  edge->dst_node = dst_node;
  edge->edge_id = edge_id;
  if (properties)
    edge->properties = *properties;

  errcode =
      graph->ops->create_edge(edge, src_node, dst_node, edge_id, properties);
  if (errcode != CL_SUCCESS) {
    goto ERROR;
  }

  /* Link to graph's edge list */
  edge->next = graph->edges;
  graph->edges = edge;

  if (errcode_ret)
    *errcode_ret = CL_SUCCESS;
  return edge;

ERROR:
  if (edge)
    free(edge);
  if (errcode_ret)
    *errcode_ret = errcode;
  return NULL;
}
POsym(clCreateWorkGraphEdgeFORMOSA)

    CL_API_ENTRY cl_int CL_API_CALL
    POname(clRetainWorkGraphFORMOSA)(cl_work_graph_formosa graph) {
  POCL_RETURN_ERROR_COND((graph == NULL), CL_INVALID_VALUE);
  POCL_RETAIN_OBJECT(graph);
  return CL_SUCCESS;
}
POsym(clRetainWorkGraphFORMOSA)

    static cl_int pocl_work_graph_formosa_collect_mem_objs(
        cl_device_id realdev, cl_context context, cl_work_graph_formosa graph,
        cl_uint num_root_inputs,
        const cl_work_graph_root_input_formosa *root_inputs,
        pocl_buffer_migration_info **dst_migr_infos) {
  pocl_buffer_migration_info *migr_infos = NULL;
  cl_int err;

  /* 1. Collect from root inputs (all read-only) */
  for (cl_uint i = 0; i < num_root_inputs; ++i) {
    if (root_inputs[i].records) {
      migr_infos = pocl_append_unique_migration_info(migr_infos,
                                                     root_inputs[i].records, 1);
    }
  }

  /* 2. Collect from all nodes in the graph */
  struct _cl_work_graph_node_formosa *node = graph->nodes;
  while (node) {
    pocl_buffer_migration_info *node_migr_infos = NULL;
    err = pocl_kernel_collect_mem_objs(realdev, context, node->kernel,
                                       node->kernel->dyn_arguments,
                                       &node_migr_infos);
    if (err != CL_SUCCESS) {
      pocl_buffer_migration_info *mi, *tmp;
      LL_FOREACH_SAFE(migr_infos, mi, tmp) {
        POname(clReleaseMemObject)(mi->buffer);
        free(mi);
      }
      return err;
    }

    /* Merge node_migr_infos into migr_infos */
    pocl_buffer_migration_info *mi, *tmp;
    LL_FOREACH_SAFE(node_migr_infos, mi, tmp) {
      migr_infos = pocl_append_unique_migration_info(migr_infos, mi->buffer,
                                                     mi->read_only);
      free(mi);
    }

    node = node->next;
  }

  *dst_migr_infos = migr_infos;
  return CL_SUCCESS;
}

CL_API_ENTRY cl_int CL_API_CALL POname(clEnqueueWorkGraphLaunchFORMOSA)(
    cl_command_queue command_queue, cl_work_graph_formosa graph,
    cl_uint num_root_inputs,
    const cl_work_graph_root_input_formosa *root_inputs,
    cl_uint num_events_in_wait_list, const cl_event *event_wait_list,
    cl_event *event) {
  cl_int errcode;
  _cl_command_node *cmd = NULL;

  POCL_RETURN_ERROR_COND((!IS_CL_OBJECT_VALID(command_queue)),
                         CL_INVALID_COMMAND_QUEUE);
  POCL_RETURN_ERROR_COND((graph == NULL), CL_INVALID_VALUE);
  POCL_RETURN_ERROR_COND((num_root_inputs > 0 && root_inputs == NULL),
                         CL_INVALID_VALUE);

  errcode = pocl_check_event_wait_list(command_queue, num_events_in_wait_list,
                                       event_wait_list);
  if (errcode != CL_SUCCESS)
    return errcode;

  if (command_queue->device->ops->run_work_graph_formosa == NULL) {
    return CL_INVALID_OPERATION;
  }

  cl_device_id realdev = pocl_real_dev(command_queue->device);
  pocl_buffer_migration_info *migr_infos = NULL;
  errcode = pocl_work_graph_formosa_collect_mem_objs(
      realdev, command_queue->context, graph, num_root_inputs, root_inputs,
      &migr_infos);
  if (errcode != CL_SUCCESS)
    return errcode;

  errcode = pocl_create_command(
      &cmd, command_queue, CL_COMMAND_GRAPH_LAUNCH_FORMOSA, event,
      num_events_in_wait_list, event_wait_list, migr_infos);
  if (errcode != CL_SUCCESS) {
    pocl_buffer_migration_info *mi, *tmp;
    LL_FOREACH_SAFE(migr_infos, mi, tmp) {
      POname(clReleaseMemObject)(mi->buffer);
      free(mi);
    }
    return errcode;
  }

  _cl_command_work_graph_launch_formosa *work_graph_launch =
      &cmd->command.work_graph_launch_formosa;
  work_graph_launch->graph = graph;
  work_graph_launch->num_root_inputs = num_root_inputs;
  work_graph_launch->root_inputs = NULL;

  POCL_RETAIN_OBJECT(graph);

  if (num_root_inputs > 0) {
    work_graph_launch->root_inputs = (cl_work_graph_root_input_formosa *)malloc(
        sizeof(cl_work_graph_root_input_formosa) * num_root_inputs);
    if (work_graph_launch->root_inputs == NULL) {
      errcode = CL_OUT_OF_HOST_MEMORY;
      goto ERROR_CLEANUP;
    }
    memcpy(work_graph_launch->root_inputs, root_inputs,
           sizeof(cl_work_graph_root_input_formosa) * num_root_inputs);
    for (cl_uint i = 0; i < num_root_inputs; ++i) {
      cl_work_graph_root_input_formosa *ri =
          &((cl_work_graph_root_input_formosa *)
                work_graph_launch->root_inputs)[i];
      if (ri->records)
        POname(clRetainMemObject)(ri->records);
    }
  }

  pocl_command_enqueue(command_queue, cmd);

  return CL_SUCCESS;

ERROR_CLEANUP:
  if (cmd) {
    cl_event ev = cmd->sync.event.event;
    pocl_update_event_failed(errcode, __FUNCTION__, __LINE__, ev,
                             "failed to setup graph launch");
  }
  return errcode;
}
POsym(clEnqueueWorkGraphLaunchFORMOSA)

    CL_API_ENTRY cl_int CL_API_CALL
    POname(clGetWorkGraphInfoFORMOSA)(cl_work_graph_formosa graph,
                                      cl_uint param_name,
                                      size_t param_value_size,
                                      void *param_value,
                                      size_t *param_value_size_ret) {
  POCL_RETURN_ERROR_COND((graph == NULL), CL_INVALID_VALUE);

  if (graph->ops->get_info == NULL) {
    return CL_INVALID_OPERATION;
  }

  return graph->ops->get_info(graph, param_name, param_value_size, param_value,
                              param_value_size_ret);
}
POsym(clGetWorkGraphInfoFORMOSA)

    cl_int pocl_formosa_release_work_graph(cl_work_graph_formosa graph) {
  if (graph == NULL)
    return CL_SUCCESS;

  int new_ref;
  POCL_RELEASE_OBJECT(graph, new_ref);

  if (new_ref == 0) {
    if (graph->ops->free_graph) {
      graph->ops->free_graph(graph);
    }

    /* Release node wrappers */
    struct _cl_work_graph_node_formosa *node = graph->nodes;
    while (node) {
      struct _cl_work_graph_node_formosa *next = node->next;
      POname(clReleaseKernel)(node->kernel);
      free(node);
      node = next;
    }

    /* Release edge wrappers */
    struct _cl_work_graph_edge_formosa *edge = graph->edges;
    while (edge) {
      struct _cl_work_graph_edge_formosa *next = edge->next;
      free(edge);
      edge = next;
    }

    POname(clReleaseContext)(graph->context);
    free(graph);
  }

  return CL_SUCCESS;
}

CL_API_ENTRY cl_int CL_API_CALL
POname(clReleaseWorkGraphFORMOSA)(cl_work_graph_formosa graph) {
  POCL_RETURN_ERROR_COND((graph == NULL), CL_INVALID_VALUE);
  return pocl_formosa_release_work_graph(graph);
}
POsym(clReleaseWorkGraphFORMOSA)
