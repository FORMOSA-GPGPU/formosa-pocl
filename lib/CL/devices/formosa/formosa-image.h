#ifndef FORMOSA_IMAGE_H
#define FORMOSA_IMAGE_H

#include "pocl_cl.h"

cl_int pocl_formosa_exec_image_command(_cl_command_node *node);
cl_int pocl_formosa_get_mapping_ptr(void *data, pocl_mem_identifier *mem,
                                    cl_mem image, mem_mapping_t *map);

cl_int pocl_formosa_copy_image_rect(void *data, cl_mem src, cl_mem dst,
                                    pocl_mem_identifier *src_mem,
                                    pocl_mem_identifier *dst_mem,
                                    const size_t *src_origin,
                                    const size_t *dst_origin,
                                    const size_t *region);
cl_int pocl_formosa_write_image_rect(void *data, cl_mem image,
                                     pocl_mem_identifier *dst_mem,
                                     const void *src_host,
                                     pocl_mem_identifier *src_mem,
                                     const size_t *origin, const size_t *region,
                                     size_t src_row_pitch,
                                     size_t src_slice_pitch, size_t src_offset);
cl_int pocl_formosa_read_image_rect(
    void *data, cl_mem image, pocl_mem_identifier *src_mem, void *dst_host,
    pocl_mem_identifier *dst_mem, const size_t *origin, const size_t *region,
    size_t dst_row_pitch, size_t dst_slice_pitch, size_t dst_offset);
cl_int pocl_formosa_map_image(void *data, pocl_mem_identifier *mem,
                              cl_mem image, mem_mapping_t *map);
cl_int pocl_formosa_unmap_image(void *data, pocl_mem_identifier *mem,
                                cl_mem image, mem_mapping_t *map);
cl_int pocl_formosa_fill_image(void *data, cl_mem image,
                               pocl_mem_identifier *mem, const size_t *origin,
                               const size_t *region, cl_uint4 orig_pixel,
                               pixel_t fill_pixel, size_t pixel_size);

#endif
