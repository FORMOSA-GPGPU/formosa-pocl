#include <CL/cl.h>
#include <CL/cl_ext.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "CL/cl_formosa_stack_remap.h"
#include "config.h"

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
  if (err != CL_SUCCESS) {
    printf("[FAIL] clGetDeviceIDs failed: %d\n", err);
    return 1;
  }

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
      printf("[FAIL] unable to size device %u extension string: %d\n", i,
             err);
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
      printf("[FAIL] unable to read device %u extension string: %d\n", i,
             err);
      free(extensions);
      failed = 1;
      continue;
    }

    const unsigned stack_count =
        count_extension(extensions, "cl_formosa_stack_remap");
    if (stack_count > 0)
      ++formosa_devices;
    if (stack_count > 1) {
      printf("[FAIL] device %u stack-remap extension count: %u\n", i,
             stack_count);
      failed = 1;
    }
    free(extensions);
  }
  free(devices);
  if (formosa_devices == 0)
    printf("[SKIP] no Formosa device available for device-string check\n");
  return failed;
}

int main() {
  cl_platform_id platform;
  cl_uint num_platforms;
  cl_int err;

  printf("Starting Formosa extension discovery test...\n");

  err = clGetPlatformIDs(1, &platform, &num_platforms);
  if (err != CL_SUCCESS || num_platforms == 0) {
    printf("Failed to find OpenCL platform. err = %d\n", err);
    /*
     * Note: If PoCL is the only platform and it has no drivers enabled,
     * this might fail with CL_PLATFORM_NOT_FOUND_KHR.
     */
    return 1;
  }

  clSetKernelStackRemapFORMOSA_fn set_stack_remap =
      (clSetKernelStackRemapFORMOSA_fn)clGetExtensionFunctionAddressForPlatform(
          platform, "clSetKernelStackRemapFORMOSA");
  if (set_stack_remap == NULL) {
    printf("[FAIL] clSetKernelStackRemapFORMOSA NOT found\n");
    return 1;
  }
  if (set_stack_remap(NULL, CL_TRUE) != CL_INVALID_KERNEL) {
    printf("[FAIL] clSetKernelStackRemapFORMOSA did not reject NULL kernel\n");
    return 1;
  }

  if (check_device_extension_strings(platform) != 0)
    return 1;

  printf(
      "Test PASSED: Formosa resolver and device extension discovery are "
      "consistent\n");
  return 0;
}
