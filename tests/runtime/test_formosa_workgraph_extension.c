#include <CL/cl.h>
#include <stdio.h>

#include "CL/cl_formosa_work_graph.h"

int main(void) {
  cl_platform_id platform;
  cl_uint num_platforms;
  cl_int err = clGetPlatformIDs(1, &platform, &num_platforms);
  if (err != CL_SUCCESS || num_platforms == 0) {
    printf("Failed to find OpenCL platform. err = %d\n", err);
    return 1;
  }

  const char *names[] = {
      "clCreateWorkGraphFORMOSA",     "clCreateWorkGraphKernelNodeFORMOSA",
      "clCreateWorkGraphEdgeFORMOSA", "clEnqueueWorkGraphLaunchFORMOSA",
      "clGetWorkGraphInfoFORMOSA",    "clRetainWorkGraphFORMOSA",
      "clReleaseWorkGraphFORMOSA"};

  int missing = 0;
  for (unsigned i = 0; i < sizeof(names) / sizeof(names[0]); ++i) {
    void *function =
        clGetExtensionFunctionAddressForPlatform(platform, names[i]);
    if (function == NULL) {
      printf("[FAIL] %s NOT found\n", names[i]);
      ++missing;
    } else {
      printf("[OK]   %s found at %p\n", names[i], function);
    }
  }

  if (missing != 0) {
    printf("Test FAILED: %d WorkGraph functions missing\n", missing);
    return 1;
  }

  printf("Test PASSED: all WorkGraph functions found at platform level\n");
  return 0;
}
