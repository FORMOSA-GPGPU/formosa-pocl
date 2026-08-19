#include <CL/cl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "CL/cl_formosa_work_graph.h"

static unsigned count_extension(const char *extensions, const char *name) {
  const size_t name_length = strlen(name);
  unsigned count = 0;
  const char *token = extensions;
  while (*token != '\0') {
    while (*token == ' ')
      ++token;
    const char *end = token;
    while (*end != '\0' && *end != ' ')
      ++end;
    if ((size_t)(end - token) == name_length &&
        memcmp(token, name, name_length) == 0)
      ++count;
    token = end;
  }
  return count;
}

static int check_device_extension_strings(cl_platform_id platform) {
  cl_uint num_devices = 0;
  cl_int err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 0, NULL,
                              &num_devices);
  if (err == CL_DEVICE_NOT_FOUND || num_devices == 0) {
    printf("[SKIP] no OpenCL devices available for extension-string check\n");
    return 0;
  }
  if (err != CL_SUCCESS)
    return 1;

  cl_device_id *devices =
      (cl_device_id *)malloc(sizeof(cl_device_id) * num_devices);
  if (devices == NULL)
    return 1;
  err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, num_devices, devices,
                       NULL);
  if (err != CL_SUCCESS) {
    free(devices);
    return 1;
  }

  int failed = 0;
  unsigned formosa_devices = 0;
  for (cl_uint i = 0; i < num_devices; ++i) {
    size_t size = 0;
    err = clGetDeviceInfo(devices[i], CL_DEVICE_EXTENSIONS, 0, NULL, &size);
    if (err != CL_SUCCESS || size == 0) {
      failed = 1;
      continue;
    }
    char *extensions = (char *)malloc(size);
    if (extensions == NULL) {
      failed = 1;
      continue;
    }
    err = clGetDeviceInfo(devices[i], CL_DEVICE_EXTENSIONS, size, extensions,
                          NULL);
    if (err != CL_SUCCESS) {
      free(extensions);
      failed = 1;
      continue;
    }
    const unsigned stack_count =
        count_extension(extensions, "cl_formosa_stack_remap");
    const unsigned graph_count =
        count_extension(extensions, "cl_formosa_work_graph");
    if (stack_count > 0) {
      ++formosa_devices;
      if (stack_count > 1 || graph_count != 1) {
        printf("[FAIL] Formosa device %u extension counts: stack=%u graph=%u\n",
               i, stack_count, graph_count);
        failed = 1;
      }
    } else if (graph_count != 0) {
      printf("[FAIL] non-Formosa device %u exposes cl_formosa_work_graph\n", i);
      failed = 1;
    }
    free(extensions);
  }
  free(devices);
  if (formosa_devices == 0)
    printf("[SKIP] no Formosa device available for device-string check\n");
  return failed;
}

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

  if (check_device_extension_strings(platform) != 0)
    return 1;

  printf("Test PASSED: WorkGraph resolver and device extension string agree\n");
  return 0;
}
