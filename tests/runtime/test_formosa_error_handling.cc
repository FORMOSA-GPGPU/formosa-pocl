/* Verify Formosa error paths fail the event (not abort / fake COMPLETE).
 * CASE1: huge __local → occupancy fail → negative event status; process alive.
 * Requires live FORMOSA daemon (AGENT_SOCKET_PATH) and installed PoCL ICD.
 */

#include <CL/cl.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

static const char *kLocalKernel = R"CLC(
__kernel void use_local(__local char *buf, __global int *out) {
  out[get_global_id(0)] = (int)buf[0];
}
)CLC";

static const char *kOkKernel = R"CLC(
__kernel void fill(__global int *out) {
  out[get_global_id(0)] = (int)get_global_id(0);
}
)CLC";

static int check_api(cl_int err, const char *what) {
  if (err != CL_SUCCESS) {
    std::fprintf(stderr, "HOST_API_ERROR %s: %d\n", what, err);
    return 1;
  }
  return 0;
}

int main() {
  cl_int err = CL_SUCCESS;
  cl_platform_id platform = nullptr;
  cl_uint nplat = 0;
  err = clGetPlatformIDs(1, &platform, &nplat);
  if (err != CL_SUCCESS || nplat == 0) {
    std::fprintf(stderr, "FAIL: no OpenCL platform (err=%d)\n", err);
    return 2;
  }

  cl_device_id device = nullptr;
  cl_uint ndev = 0;
  err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, &ndev);
  if (err != CL_SUCCESS || ndev == 0) {
    std::fprintf(stderr, "FAIL: no GPU device (err=%d)\n", err);
    return 2;
  }

  char name[256] = {};
  clGetDeviceInfo(device, CL_DEVICE_NAME, sizeof(name), name, nullptr);
  std::printf("Device: %s\n", name);

  cl_ulong local_mem = 0;
  clGetDeviceInfo(device, CL_DEVICE_LOCAL_MEM_SIZE, sizeof(local_mem),
                  &local_mem, nullptr);
  std::printf("CL_DEVICE_LOCAL_MEM_SIZE=%llu\n", (unsigned long long)local_mem);

  cl_context ctx = clCreateContext(nullptr, 1, &device, nullptr, nullptr, &err);
  if (check_api(err, "clCreateContext"))
    return 2;

  cl_command_queue q = clCreateCommandQueue(ctx, device, 0, &err);
  if (check_api(err, "clCreateCommandQueue"))
    return 2;

  /* ---------- CASE1: occupancy failure must fail the event ---------- */
  const char *src = kLocalKernel;
  cl_program prog = clCreateProgramWithSource(ctx, 1, &src, nullptr, &err);
  if (check_api(err, "clCreateProgramWithSource"))
    return 2;
  err = clBuildProgram(prog, 1, &device, nullptr, nullptr, nullptr);
  if (err != CL_SUCCESS) {
    size_t log_size = 0;
    clGetProgramBuildInfo(prog, device, CL_PROGRAM_BUILD_LOG, 0, nullptr,
                          &log_size);
    std::vector<char> log(log_size + 1);
    clGetProgramBuildInfo(prog, device, CL_PROGRAM_BUILD_LOG, log_size,
                          log.data(), nullptr);
    std::fprintf(stderr, "FAIL: build: %s\n", log.data());
    return 2;
  }

  cl_kernel kernel = clCreateKernel(prog, "use_local", &err);
  if (check_api(err, "clCreateKernel"))
    return 2;

  const size_t huge_local = (size_t)local_mem + 4096;
  std::printf("CASE1: __local size=%zu (device local=%llu)\n", huge_local,
              (unsigned long long)local_mem);
  err = clSetKernelArg(kernel, 0, huge_local, nullptr);
  if (check_api(err, "clSetKernelArg local"))
    return 2;

  cl_mem out =
      clCreateBuffer(ctx, CL_MEM_WRITE_ONLY, sizeof(int) * 4, nullptr, &err);
  if (check_api(err, "clCreateBuffer"))
    return 2;
  err = clSetKernelArg(kernel, 1, sizeof(cl_mem), &out);
  if (check_api(err, "clSetKernelArg out"))
    return 2;

  const size_t gws = 4;
  const size_t lws = 4;
  cl_event ev = nullptr;
  std::printf("CASE1: enqueue NDRange with event\n");
  err = clEnqueueNDRangeKernel(q, kernel, 1, nullptr, &gws, &lws, 0, nullptr,
                               &ev);
  std::printf("CASE1: clEnqueueNDRangeKernel returned %d\n", err);
  if (err != CL_SUCCESS || ev == nullptr) {
    std::fprintf(stderr,
                 "FAIL: expected enqueue to succeed and produce an event "
                 "(err=%d, ev=%p); Formosa defers occupancy failure to "
                 "event status\n",
                 err, (void *)ev);
    if (ev)
      clReleaseEvent(ev);
    return 2;
  }

  err = clFinish(q);
  std::printf("CASE1: clFinish returned %d\n", err);

  cl_int ev_status = 0;
  err = clGetEventInfo(ev, CL_EVENT_COMMAND_EXECUTION_STATUS, sizeof(ev_status),
                       &ev_status, nullptr);
  if (check_api(err, "clGetEventInfo"))
    return 2;
  std::printf("CASE1: event execution status = %d\n", ev_status);

  int case1_ok = 0;
  if (ev_status < 0) {
    std::printf(
        "CASE1: PASS — event ended as failed (status=%d), process alive\n",
        ev_status);
    case1_ok = 1;
  } else {
    std::printf("CASE1: FAIL — expected negative event status, got %d (fake "
                "COMPLETE?)\n",
                ev_status);
  }

  clReleaseEvent(ev);
  clReleaseMemObject(out);
  clReleaseKernel(kernel);
  clReleaseProgram(prog);

  /* ---------- CASE2: good kernel still works after failed command ---------- */
  src = kOkKernel;
  prog = clCreateProgramWithSource(ctx, 1, &src, nullptr, &err);
  if (check_api(err, "clCreateProgramWithSource ok"))
    return 2;
  err = clBuildProgram(prog, 1, &device, nullptr, nullptr, nullptr);
  if (check_api(err, "clBuildProgram ok"))
    return 2;
  kernel = clCreateKernel(prog, "fill", &err);
  if (check_api(err, "clCreateKernel fill"))
    return 2;
  out = clCreateBuffer(ctx, CL_MEM_WRITE_ONLY, sizeof(int) * 4, nullptr, &err);
  if (check_api(err, "clCreateBuffer ok"))
    return 2;
  err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &out);
  if (check_api(err, "clSetKernelArg fill"))
    return 2;

  cl_event ev2 = nullptr;
  err = clEnqueueNDRangeKernel(q, kernel, 1, nullptr, &gws, &lws, 0, nullptr,
                               &ev2);
  std::printf("CASE2: clEnqueueNDRangeKernel returned %d\n", err);
  if (err != CL_SUCCESS || ev2 == nullptr) {
    std::fprintf(stderr, "CASE2: FAIL enqueue (err=%d, ev=%p)\n", err,
                 (void *)ev2);
    if (ev2)
      clReleaseEvent(ev2);
    return 2;
  }
  err = clFinish(q);
  std::printf("CASE2: clFinish returned %d\n", err);
  cl_int ev2_status = 0;
  err = clGetEventInfo(ev2, CL_EVENT_COMMAND_EXECUTION_STATUS,
                       sizeof(ev2_status), &ev2_status, nullptr);
  if (check_api(err, "clGetEventInfo case2"))
    return 2;
  std::printf("CASE2: event execution status = %d\n", ev2_status);

  int case2_ok = (ev2_status == CL_COMPLETE);
  if (case2_ok)
    std::printf("CASE2: PASS — subsequent good kernel completes\n");
  else
    std::printf("CASE2: FAIL — good kernel status=%d\n", ev2_status);

  /* verify device result for good kernel */
  int host[4] = {};
  clEnqueueReadBuffer(q, out, CL_TRUE, 0, sizeof(host), host, 0, nullptr,
                      nullptr);
  for (int i = 0; i < 4; i++) {
    if (host[i] != i) {
      std::printf("CASE2: FAIL — host[%d]=%d expected %d\n", i, host[i], i);
      case2_ok = 0;
    }
  }

  clReleaseEvent(ev2);
  clReleaseMemObject(out);
  clReleaseKernel(kernel);
  clReleaseProgram(prog);
  clReleaseCommandQueue(q);
  clReleaseContext(ctx);

  if (case1_ok && case2_ok) {
    std::printf("SUCCESS: errors fail the event; process stays alive\n");
    return 0;
  }
  std::printf("FAIL: error-handling contract not met\n");
  return 1;
}
