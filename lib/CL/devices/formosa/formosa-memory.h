#ifndef FORMOSA_MEMORY_H
#define FORMOSA_MEMORY_H

#include <formosa-hal/formosa-hal.h>

#include "pocl.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
  uint64_t buf_address;
  uint64_t buf_size;
  uint64_t msg_id;
} formosa_buffer_data_t;

typedef struct {
  uint64_t src_addr;
  uint64_t dst_addr;
} formosa_memory_copy_addresses_t;

/* Propagate HAL transport failure to PoCL device availability. */
void pocl_formosa_mark_unavailable(void);

/* Resolve a PoCL memory identifier into a validated Formosa device range. */
cl_int formosa_memory_resolve_buffer_address(const pocl_mem_identifier *mem_id,
                                             size_t offset, size_t size,
                                             uint64_t *device_addr);

/* Resolve both ranges used by a device-to-device copy. */
cl_int formosa_memory_resolve_copy_addresses(
    const pocl_mem_identifier *dst_mem_id,
    const pocl_mem_identifier *src_mem_id, size_t dst_offset, size_t src_offset,
    size_t size, formosa_memory_copy_addresses_t *addresses);

/* Submit an asynchronous copy through the Formosa firmware command stream. */
cl_int formosa_memory_submit_copy(MemoryDomain src_domain, uint64_t src_addr,
                                  MemoryDomain dst_domain, uint64_t dst_addr,
                                  size_t size,
                                  FsaMemoryCopyCompletion *completion);

/* Submit and wait for a copy, returning the corresponding OpenCL error. */
cl_int formosa_memory_copy(MemoryDomain src_domain, uint64_t src_addr,
                           MemoryDomain dst_domain, uint64_t dst_addr,
                           size_t size);

/* Convert a HAL copy result to an OpenCL status for asynchronous callers. */
cl_int formosa_memory_copy_result_to_cl(MemoryCopyResult result);

#ifdef __cplusplus
}
#endif

#endif  // FORMOSA_MEMORY_H
