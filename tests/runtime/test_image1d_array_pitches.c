// SPDX-FileCopyrightText: 2026 CASLab, National Cheng Kung University
//
// SPDX-License-Identifier: MIT

#include "poclu.h"
#include <stdio.h>
#include <string.h>

int main(void) {
  cl_context context;
  cl_device_id device;
  cl_command_queue queue;
  CHECK_CL_ERROR(poclu_get_any_device(&context, &device, &queue));
  cl_bool supported;
  CHECK_CL_ERROR(clGetDeviceInfo(device, CL_DEVICE_IMAGE_SUPPORT,
                                sizeof(supported), &supported, NULL));
  if (!supported) {
    clReleaseCommandQueue(queue);
    clReleaseContext(context);
    return 77;
  }

  const cl_image_format format = {CL_RGBA, CL_UNSIGNED_INT8};
  cl_image_desc desc = {0};
  desc.image_type = CL_MEM_OBJECT_IMAGE1D_ARRAY;
  desc.image_width = 2;
  desc.image_array_size = 3;
  const size_t origin[3] = {0, 0, 0}, region[3] = {2, 3, 1};
  unsigned char input[48], packed[24], output[48];
  memset(input, 0xa5, sizeof(input));
  for (size_t layer = 0; layer < 3; ++layer)
    for (size_t x = 0; x < 8; ++x)
      input[layer * 16 + x] = packed[layer * 8 + x] = layer * 17 + x;

  cl_int err;
  cl_mem image = clCreateImage(context, CL_MEM_READ_WRITE, &format, &desc,
                               NULL, &err);
  CHECK_CL_ERROR(err);
  CHECK_CL_ERROR(clEnqueueWriteImage(queue, image, CL_TRUE, origin, region,
                                     8, 16, input, 0, NULL, NULL));
  CHECK_CL_ERROR(clEnqueueReadImage(queue, image, CL_TRUE, origin, region,
                                    0, 0, output, 0, NULL, NULL));
  int failed = memcmp(output, packed, sizeof(packed)) != 0;
  if (failed) fprintf(stderr, "1D array padded write mismatch\n");
  CHECK_CL_ERROR(clReleaseMemObject(image));

  unsigned char backing[72];
  memset(backing, 0xa5, sizeof(backing));
  for (size_t layer = 0; layer < 3; ++layer)
    memcpy(backing + layer * 24, packed + layer * 8, 8);
  desc.image_row_pitch = 8;
  desc.image_slice_pitch = 24;
  image = clCreateImage(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                        &format, &desc, backing, &err);
  CHECK_CL_ERROR(err);
  memset(output, 0xa5, sizeof(output));
  const size_t partial_origin[3] = {0, 1, 0};
  const size_t partial_region[3] = {2, 2, 1};
  CHECK_CL_ERROR(clEnqueueReadImage(queue, image, CL_TRUE, partial_origin,
                                    partial_region, 8, 16, output, 0, NULL, NULL));
  for (size_t i = 0; i < sizeof(output); ++i) {
    unsigned char expected = (i / 16 < 2 && i % 16 < 8)
        ? packed[(i / 16 + 1) * 8 + i % 16] : 0xa5;
    if (output[i] != expected) {
      fprintf(stderr, "1D array padded read mismatch at byte %zu\n", i);
      failed = 1;
      break;
    }
  }
  CHECK_CL_ERROR(clReleaseMemObject(image));
  CHECK_CL_ERROR(clReleaseCommandQueue(queue));
  CHECK_CL_ERROR(clReleaseContext(context));
  return failed;
}
