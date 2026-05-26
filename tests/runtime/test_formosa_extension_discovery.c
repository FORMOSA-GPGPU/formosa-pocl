#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <CL/cl.h>
#include <CL/cl_ext.h>
#include "CL/cl_fsa_work_graph.h"

int main() {
    cl_platform_id platform;
    cl_uint num_platforms;
    cl_int err;

    printf("Starting Phase 0 extension discovery test (platform-level only)...\n");

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

    const char *names[] = {
        "clfsaCreateGraph",
        "clfsaCreateGraphKernelNode",
        "clfsaCreateGraphEdge",
        "clfsaEnqueueGraphLaunch",
        "clfsaGetGraphInfo",
        "clfsaRetainGraph",
        "clfsaReleaseGraph"
    };

    int missing = 0;
    for (int i = 0; i < 7; ++i) {
        void *p = clGetExtensionFunctionAddressForPlatform(platform, names[i]);
        if (!p) {
            printf("[FAIL] %s NOT found via clGetExtensionFunctionAddressForPlatform\n", names[i]);
            missing++;
        } else {
            printf("[OK]   %s found at %p\n", names[i], p);
        }
    }

    if (missing > 0) {
        printf("Test FAILED: %d extension functions missing\n", missing);
        return 1;
    }

    printf("Test PASSED: All extension functions found successfully at platform level\n");
    return 0;
}
