#include <CL/cl.h>
#include <CL/cl_ext.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "CL/cl_formosa_stack_remap.h"

int main() {
  cl_platform_id platform;
  cl_uint num_platforms;
  cl_int err;

  printf(
      "Starting Formosa extension discovery test (platform-level only)...\n");

  /*
   * Layer A test:
   * - Only use platform-level APIs.
   * - Do NOT initialize any device (clGetDeviceIDs is NOT called).
   * - This test should run without the Formosa daemon or a real device.
   */

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

  printf(
      "Test PASSED: Formosa stack-remap function found successfully at "
      "platform level\n");
  return 0;
}
