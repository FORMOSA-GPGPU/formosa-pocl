#include "formosa-image.h"

#include <stdlib.h>
#include <string.h>

#include "formosa-memory.h"
#include "pocl_util.h"

static size_t image_row_stride(cl_mem image) {
  return image->type == CL_MEM_OBJECT_IMAGE1D_ARRAY ? image->image_slice_pitch
                                                    : image->image_row_pitch;
}

cl_int pocl_formosa_write_image_rect(
    void *data, cl_mem image, pocl_mem_identifier *dst_mem,
    const void *src_host, pocl_mem_identifier *src_mem, const size_t *origin,
    const size_t *region, size_t src_row_pitch, size_t src_slice_pitch,
    size_t src_offset) {
  (void)data;
  size_t row_bytes = region[0] * image->image_elem_size * image->image_channels;
  if (src_row_pitch == 0) src_row_pitch = row_bytes;
  if (image->type == CL_MEM_OBJECT_IMAGE1D_ARRAY)
    src_row_pitch = src_slice_pitch ? src_slice_pitch : src_row_pitch;
  if (src_slice_pitch == 0) src_slice_pitch = src_row_pitch * region[1];
  size_t dst_offset =
      origin[0] * image->image_elem_size * image->image_channels +
      origin[1] * image_row_stride(image) +
      origin[2] * image->image_slice_pitch;
  return formosa_memory_copy_rows(
      dst_mem, NULL, dst_offset, image_row_stride(image),
      image->image_slice_pitch, src_mem, src_host, src_offset, src_row_pitch,
      src_slice_pitch, region, row_bytes);
}

cl_int pocl_formosa_read_image_rect(
    void *data, cl_mem image, pocl_mem_identifier *src_mem, void *dst_host,
    pocl_mem_identifier *dst_mem, const size_t *origin, const size_t *region,
    size_t dst_row_pitch, size_t dst_slice_pitch, size_t dst_offset) {
  (void)data;
  size_t row_bytes = region[0] * image->image_elem_size * image->image_channels;
  if (dst_row_pitch == 0) dst_row_pitch = row_bytes;
  if (image->type == CL_MEM_OBJECT_IMAGE1D_ARRAY)
    dst_row_pitch = dst_slice_pitch ? dst_slice_pitch : dst_row_pitch;
  if (dst_slice_pitch == 0) dst_slice_pitch = dst_row_pitch * region[1];
  size_t src_offset =
      origin[0] * image->image_elem_size * image->image_channels +
      origin[1] * image_row_stride(image) +
      origin[2] * image->image_slice_pitch;
  return formosa_memory_copy_rows(dst_mem, dst_host, dst_offset, dst_row_pitch,
                                  dst_slice_pitch, src_mem, NULL, src_offset,
                                  image_row_stride(image),
                                  image->image_slice_pitch, region, row_bytes);
}

cl_int pocl_formosa_copy_image_rect(void *data, cl_mem src, cl_mem dst,
                                    pocl_mem_identifier *src_mem,
                                    pocl_mem_identifier *dst_mem,
                                    const size_t *src_origin,
                                    const size_t *dst_origin,
                                    const size_t *region) {
  (void)data;
  size_t pixel_bytes = src->image_elem_size * src->image_channels;
  size_t src_offset = src_origin[0] * pixel_bytes +
                      src_origin[1] * image_row_stride(src) +
                      src_origin[2] * src->image_slice_pitch;
  size_t dst_offset = dst_origin[0] * pixel_bytes +
                      dst_origin[1] * image_row_stride(dst) +
                      dst_origin[2] * dst->image_slice_pitch;
  return formosa_memory_copy_rows(
      dst_mem, NULL, dst_offset, image_row_stride(dst), dst->image_slice_pitch,
      src_mem, NULL, src_offset, image_row_stride(src), src->image_slice_pitch,
      region, region[0] * pixel_bytes);
}

cl_int pocl_formosa_map_image(void *data, pocl_mem_identifier *mem,
                              cl_mem image, mem_mapping_t *map) {
  if (map->map_flags & CL_MAP_WRITE_INVALIDATE_REGION) return CL_SUCCESS;
  return pocl_formosa_read_image_rect(data, image, mem, map->host_ptr, NULL,
                                      map->origin, map->region, map->row_pitch,
                                      map->slice_pitch, 0);
}

cl_int pocl_formosa_unmap_image(void *data, pocl_mem_identifier *mem,
                                cl_mem image, mem_mapping_t *map) {
  if (!(map->map_flags & (CL_MAP_WRITE | CL_MAP_WRITE_INVALIDATE_REGION)))
    return CL_SUCCESS;
  return pocl_formosa_write_image_rect(data, image, mem, map->host_ptr, NULL,
                                       map->origin, map->region, map->row_pitch,
                                       map->slice_pitch, 0);
}

cl_int pocl_formosa_fill_image(void *data, cl_mem image,
                               pocl_mem_identifier *mem, const size_t *origin,
                               const size_t *region, cl_uint4 orig_pixel,
                               pixel_t fill_pixel, size_t pixel_size) {
  (void)data;
  (void)orig_pixel;
  size_t row_bytes = region[0] * pixel_size;
  char *row = malloc(row_bytes);
  if (row == NULL) return CL_OUT_OF_HOST_MEMORY;
  for (size_t x = 0; x < region[0]; ++x)
    memcpy(row + x * pixel_size, fill_pixel, pixel_size);
  size_t dst_offset = origin[0] * pixel_size +
                      origin[1] * image_row_stride(image) +
                      origin[2] * image->image_slice_pitch;
  cl_int err = formosa_memory_copy_rows(
      mem, NULL, dst_offset, image_row_stride(image), image->image_slice_pitch,
      NULL, row, 0, 0, 0, region, row_bytes);
  free(row);
  return err;
}

cl_int pocl_formosa_exec_image_command(_cl_command_node *node) {
  cl_device_id dev = node->device;
  _cl_command_t *cmd = &node->command;
  unsigned mem_id = dev->global_mem_id;
  switch (node->type) {
    case CL_COMMAND_READ_IMAGE:
    case CL_COMMAND_COPY_IMAGE_TO_BUFFER:
      return pocl_formosa_read_image_rect(
          dev->data, cmd->read_image.src,
          &POCL_MEM_BS(cmd->read_image.src)->device_ptrs[mem_id],
          cmd->read_image.dst_host_ptr,
          cmd->read_image.dst == NULL
              ? NULL
              : &POCL_MEM_BS(cmd->read_image.dst)->device_ptrs[mem_id],
          cmd->read_image.origin, cmd->read_image.region,
          cmd->read_image.dst_row_pitch, cmd->read_image.dst_slice_pitch,
          cmd->read_image.dst_offset);
    case CL_COMMAND_WRITE_IMAGE:
    case CL_COMMAND_COPY_BUFFER_TO_IMAGE:
      return pocl_formosa_write_image_rect(
          dev->data, cmd->write_image.dst,
          &POCL_MEM_BS(cmd->write_image.dst)->device_ptrs[mem_id],
          cmd->write_image.src_host_ptr,
          cmd->write_image.src == NULL
              ? NULL
              : &POCL_MEM_BS(cmd->write_image.src)->device_ptrs[mem_id],
          cmd->write_image.origin, cmd->write_image.region,
          cmd->write_image.src_row_pitch, cmd->write_image.src_slice_pitch,
          cmd->write_image.src_offset);
    case CL_COMMAND_COPY_IMAGE:
      return pocl_formosa_copy_image_rect(
          dev->data, cmd->copy_image.src, cmd->copy_image.dst,
          &POCL_MEM_BS(cmd->copy_image.src)->device_ptrs[mem_id],
          &POCL_MEM_BS(cmd->copy_image.dst)->device_ptrs[mem_id],
          cmd->copy_image.src_origin, cmd->copy_image.dst_origin,
          cmd->copy_image.region);
    case CL_COMMAND_FILL_IMAGE:
      return pocl_formosa_fill_image(
          dev->data, cmd->fill_image.dst,
          &cmd->fill_image.dst->device_ptrs[mem_id], cmd->fill_image.origin,
          cmd->fill_image.region, cmd->fill_image.orig_pixel,
          cmd->fill_image.fill_pixel, cmd->fill_image.pixel_size);
    case CL_COMMAND_MAP_IMAGE:
      return pocl_formosa_map_image(dev->data,
                                    &cmd->map.buffer->device_ptrs[mem_id],
                                    cmd->map.buffer, cmd->map.mapping);
    case CL_COMMAND_UNMAP_MEM_OBJECT:
      return pocl_formosa_unmap_image(
          dev->data, &POCL_MEM_BS(cmd->unmap.buffer)->device_ptrs[mem_id],
          cmd->unmap.buffer, cmd->unmap.mapping);
    default:
      return CL_INVALID_OPERATION;
  }
}
