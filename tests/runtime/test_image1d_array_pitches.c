// SPDX-FileCopyrightText: 2026 CASLab, National Cheng Kung University
//
// SPDX-License-Identifier: MIT

#include <stdio.h>
#include <string.h>

#include "poclu.h"

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
  cl_mem image =
      clCreateImage(context, CL_MEM_READ_WRITE, &format, &desc, NULL, &err);
  CHECK_CL_ERROR(err);
  CHECK_CL_ERROR(clEnqueueWriteImage(queue, image, CL_TRUE, origin, region, 8,
                                     16, input, 0, NULL, NULL));
  CHECK_CL_ERROR(clEnqueueReadImage(queue, image, CL_TRUE, origin, region, 0, 0,
                                    output, 0, NULL, NULL));
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
                                    partial_region, 8, 16, output, 0, NULL,
                                    NULL));
  for (size_t i = 0; i < sizeof(output); ++i) {
    unsigned char expected =
        (i / 16 < 2 && i % 16 < 8) ? packed[(i / 16 + 1) * 8 + i % 16] : 0xa5;
    if (output[i] != expected) {
      fprintf(stderr, "1D array padded read mismatch at byte %zu\n", i);
      failed = 1;
      break;
    }
  }
  CHECK_CL_ERROR(clReleaseMemObject(image));

  unsigned char mapped_backing[48], expected_backing[48];
  memcpy(mapped_backing, input, sizeof(mapped_backing));
  memcpy(expected_backing, input, sizeof(expected_backing));
  desc.image_slice_pitch = 16;
  image = clCreateImage(context, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR,
                        &format, &desc, mapped_backing, &err);
  CHECK_CL_ERROR(err);
  const size_t map_origin[3] = {0, 2, 0}, map_region[3] = {2, 1, 1};
  size_t row_pitch, slice_pitch;
  unsigned char *mapped = clEnqueueMapImage(
      queue, image, CL_TRUE, CL_MAP_READ | CL_MAP_WRITE, map_origin, map_region,
      &row_pitch, &slice_pitch, 0, NULL, NULL, &err);
  CHECK_CL_ERROR(err);
  if (mapped != mapped_backing + 32 || row_pitch != 8 || slice_pitch != 16 ||
      memcmp(mapped, packed + 16, 8) != 0) {
    fprintf(stderr, "1D array host map offset/pitch mismatch\n");
    failed = 1;
  }
  memset(mapped, 0x5a, 8);
  memset(expected_backing + 32, 0x5a, 8);
  CHECK_CL_ERROR(clEnqueueUnmapMemObject(queue, image, mapped, 0, NULL, NULL));
  CHECK_CL_ERROR(clFinish(queue));
  if (memcmp(mapped_backing, expected_backing, sizeof(mapped_backing)) != 0) {
    fprintf(stderr, "1D array map modified the wrong host layer\n");
    failed = 1;
  }
  CHECK_CL_ERROR(clEnqueueReadImage(queue, image, CL_TRUE, origin, region, 0, 0,
                                    output, 0, NULL, NULL));
  if (memcmp(output, packed, 16) != 0 ||
      memcmp(output + 16, mapped_backing + 32, 8) != 0) {
    fprintf(stderr, "1D array unmap device data mismatch\n");
    failed = 1;
  }
  CHECK_CL_ERROR(clReleaseMemObject(image));

  desc.image_slice_pitch = 24;
  image = clCreateImage(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                        &format, &desc, backing, &err);
  CHECK_CL_ERROR(err);
  unsigned char buffer_data[32];
  memset(buffer_data, 0xa5, sizeof(buffer_data));
  for (size_t i = 0; i < sizeof(packed); ++i)
    buffer_data[4 + i] = packed[i] ^ 0xff;
  cl_mem buffer =
      clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                     sizeof(buffer_data), buffer_data, &err);
  CHECK_CL_ERROR(err);
  CHECK_CL_ERROR(clEnqueueCopyBufferToImage(queue, buffer, image, 4, origin,
                                            region, 0, NULL, NULL));
  CHECK_CL_ERROR(clEnqueueReadImage(queue, image, CL_TRUE, origin, region, 0, 0,
                                    output, 0, NULL, NULL));
  if (memcmp(output, buffer_data + 4, sizeof(packed)) != 0) {
    fprintf(stderr, "1D array buffer-to-image copy mismatch\n");
    failed = 1;
  }
  const unsigned char fill = 0xa5;
  CHECK_CL_ERROR(clEnqueueFillBuffer(queue, buffer, &fill, sizeof(fill), 0,
                                     sizeof(buffer_data), 0, NULL, NULL));
  CHECK_CL_ERROR(clEnqueueCopyImageToBuffer(queue, image, buffer, origin,
                                            region, 4, 0, NULL, NULL));
  CHECK_CL_ERROR(clEnqueueReadBuffer(
      queue, buffer, CL_TRUE, 0, sizeof(buffer_data), output, 0, NULL, NULL));
  if (memcmp(output, buffer_data, sizeof(buffer_data)) != 0) {
    fprintf(stderr, "1D array image-to-buffer copy mismatch\n");
    failed = 1;
  }
  CHECK_CL_ERROR(clReleaseMemObject(buffer));
  CHECK_CL_ERROR(clReleaseMemObject(image));
  CHECK_CL_ERROR(clReleaseCommandQueue(queue));
  CHECK_CL_ERROR(clReleaseContext(context));
  return failed;
}
