#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <CL/cl.h>
#include <CL/cl_fsa_work_graph.h>

#define SKIP_CODE 77

#define CHECK_ERR(err)                                                         \
  if (err != CL_SUCCESS) {                                                     \
    printf("Error: %d at line %d\n", err, __LINE__);                           \
    exit(1);                                                                   \
  }

const char *source =
    "#include <CL/cl_fsa_work_graph_device.h>\n"
    "typedef struct { uint value; } InputRecord;\n"
    "__kernel void single_node_kernel(__global uint *out) {\n"
    "    uint gid = get_global_id(0);\n"
    "    if (gid >= clfsa_get_record_count()) return;\n"
    "    InputRecord rec;\n"
    "    if (clfsa_get_record(gid, &rec, sizeof(rec)) != 0) return;\n"
    "    out[gid] = rec.value + 100;\n"
    "}\n";

typedef struct {
  uint32_t value;
} InputRecord;

int main() {
  printf("Starting FSA Work Graph Phase 1 single-node test\n");

  cl_platform_id platform;
  cl_device_id device = NULL;
  cl_int err;

  err = clGetPlatformIDs(1, &platform, NULL);
  CHECK_ERR(err);

  /* Get extension functions */
  clfsaCreateGraph_fn p_clfsaCreateGraph =
      (clfsaCreateGraph_fn)clGetExtensionFunctionAddressForPlatform(
          platform, "clfsaCreateGraph");
  clfsaCreateGraphKernelNode_fn p_clfsaCreateGraphKernelNode =
      (clfsaCreateGraphKernelNode_fn)clGetExtensionFunctionAddressForPlatform(
          platform, "clfsaCreateGraphKernelNode");
  clfsaEnqueueGraphLaunch_fn p_clfsaEnqueueGraphLaunch =
      (clfsaEnqueueGraphLaunch_fn)clGetExtensionFunctionAddressForPlatform(
          platform, "clfsaEnqueueGraphLaunch");
  clfsaReleaseGraph_fn p_clfsaReleaseGraph =
      (clfsaReleaseGraph_fn)clGetExtensionFunctionAddressForPlatform(
          platform, "clfsaReleaseGraph");

  if (!p_clfsaCreateGraph || !p_clfsaCreateGraphKernelNode ||
      !p_clfsaEnqueueGraphLaunch || !p_clfsaReleaseGraph) {
    printf("SKIP: FSA Work Graph extension functions not found\n");
    return SKIP_CODE;
  }

  cl_uint num_devices;
  err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 0, NULL, &num_devices);
  CHECK_ERR(err);
  cl_device_id *devices =
      (cl_device_id *)malloc(sizeof(cl_device_id) * num_devices);
  err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, num_devices, devices, NULL);
  CHECK_ERR(err);

  for (cl_uint i = 0; i < num_devices; i++) {
    char name[256];
    clGetDeviceInfo(devices[i], CL_DEVICE_NAME, sizeof(name), name, NULL);
    if (strstr(name, "formosa") || strstr(name, "Formosa")) {
      device = devices[i];
      printf("Found Formosa device: %s\n", name);
      break;
    }
  }
  free(devices);

  if (!device) {
    printf("SKIP: No Formosa device found\n");
    return SKIP_CODE;
  }

  cl_context context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
  CHECK_ERR(err);

  cl_command_queue queue = clCreateCommandQueue(context, device, 0, &err);
  CHECK_ERR(err);

  cl_program program =
      clCreateProgramWithSource(context, 1, &source, NULL, &err);
  CHECK_ERR(err);

  char build_options[1024];
#ifdef POCL_INSTALL_PUBLIC_HEADER_DIR
  snprintf(build_options, sizeof(build_options), "-I%s",
           POCL_INSTALL_PUBLIC_HEADER_DIR);
#else
  build_options[0] = '\0';
#endif

  err = clBuildProgram(program, 1, &device, build_options, NULL, NULL);
  if (err != CL_SUCCESS) {
    size_t log_size;
    clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, NULL,
                          &log_size);
    char *log = (char *)malloc(log_size);
    clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, log_size, log,
                          NULL);
    printf("Build Log:\n%s\n", log);
    free(log);
    CHECK_ERR(err);
  }

  cl_kernel kernel = clCreateKernel(program, "single_node_kernel", &err);
  CHECK_ERR(err);

  uint32_t records_data[] = {1, 2, 3, 4};
  cl_mem input_buf = clCreateBuffer(
      context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(records_data),
      records_data, &err);
  CHECK_ERR(err);

  uint32_t output_data[4] = {0};
  cl_mem output_buf =
      clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(output_data), NULL, &err);
  CHECK_ERR(err);

  err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &output_buf);
  CHECK_ERR(err);

  /* Phase 1: Create Graph */
  cl_fsa_graph_properties graph_props = {0, 1, 0, 4};
  cl_fsa_graph graph = p_clfsaCreateGraph(context, device, &graph_props, &err);
  CHECK_ERR(err);
  printf("Created graph\n");

  /* Create Single Node */
  cl_fsa_node_properties node_props = {
      CL_FSA_NODE_ROOT_CAPABLE, sizeof(InputRecord), CL_FSA_NODE_LAUNCH_THREAD,
      4, 1};
  size_t global_size = 4;
  size_t local_size = 1;
  cl_fsa_graph_node node = p_clfsaCreateGraphKernelNode(
      graph, kernel, 0, 1, NULL, &global_size, &local_size, &node_props, &err);
  CHECK_ERR(err);
  printf("Created single graph node\n");

  /* Create Root Input */
  cl_fsa_root_input root_input = {0, 4, input_buf, 0, sizeof(InputRecord)};

  /* Launch Graph */
  printf("Launching graph\n");
  cl_event event;
  err = p_clfsaEnqueueGraphLaunch(queue, graph, 1, &root_input, 0, NULL, &event);
  CHECK_ERR(err);

  printf("Waiting for event\n");
  err = clWaitForEvents(1, &event);
  CHECK_ERR(err);

  err = clEnqueueReadBuffer(queue, output_buf, CL_TRUE, 0, sizeof(output_data),
                            output_data, 0, NULL, NULL);
  CHECK_ERR(err);

  int pass = 1;
  for (int i = 0; i < 4; i++) {
    printf("Output[%d] = %d\n", i, output_data[i]);
    if (output_data[i] != 101 + i)
      pass = 0;
  }

  if (pass) {
    printf("Test PASSED\n");
  } else {
    printf("Test FAILED: Output mismatch\n");
    exit(1);
  }

  clReleaseEvent(event);
  p_clfsaReleaseGraph(graph);
  clReleaseKernel(kernel);
  clReleaseProgram(program);
  clReleaseMemObject(input_buf);
  clReleaseMemObject(output_buf);
  clReleaseCommandQueue(queue);
  clReleaseContext(context);

  return 0;
}
