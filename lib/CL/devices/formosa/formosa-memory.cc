#include "formosa-memory.h"

#include "pocl_debug.h"

namespace {

static cl_bool formosa_memory_get_buffer_device_address(
    const formosa_buffer_data_t *buffer, size_t offset, size_t size,
    uint64_t *device_addr) {
  if (buffer == nullptr || offset > buffer->buf_size ||
      size > buffer->buf_size - offset)
    return CL_FALSE;

  if ((uint64_t)offset > UINT64_MAX - buffer->buf_address) return CL_FALSE;

  uint64_t address = buffer->buf_address + (uint64_t)offset;
  if ((uint64_t)size > UINT64_MAX - address) return CL_FALSE;

  if (device_addr != nullptr) *device_addr = address;
  return CL_TRUE;
}

}  // namespace

cl_int formosa_memory_resolve_buffer_address(const pocl_mem_identifier *mem_id,
                                             size_t offset, size_t size,
                                             uint64_t *device_addr) {
  if (mem_id == nullptr || mem_id->mem_ptr == nullptr)
    return CL_INVALID_MEM_OBJECT;

  const formosa_buffer_data_t *buffer =
      (const formosa_buffer_data_t *)mem_id->mem_ptr;
  return formosa_memory_get_buffer_device_address(buffer, offset, size,
                                                  device_addr)
             ? CL_SUCCESS
             : CL_INVALID_VALUE;
}

cl_int formosa_memory_resolve_copy_addresses(
    const pocl_mem_identifier *dst_mem_id,
    const pocl_mem_identifier *src_mem_id, size_t dst_offset, size_t src_offset,
    size_t size, formosa_memory_copy_addresses_t *addresses) {
  if (addresses == nullptr) return CL_INVALID_VALUE;

  cl_int err = formosa_memory_resolve_buffer_address(
      src_mem_id, src_offset, size, &addresses->src_addr);
  if (err != CL_SUCCESS) return err;
  return formosa_memory_resolve_buffer_address(dst_mem_id, dst_offset, size,
                                               &addresses->dst_addr);
}

cl_int formosa_memory_copy_result_to_cl(MemoryCopyResult result) {
  switch (result) {
    case kMemoryCopyResultSuccess:
      return CL_SUCCESS;
    case kMemoryCopyResultOverlap:
      return CL_MEM_COPY_OVERLAP;
    case kMemoryCopyResultInvalidAddress:
    case kMemoryCopyResultInvalidRange:
    case kMemoryCopyResultInvalidDomainPair:
      return CL_INVALID_VALUE;
    case kMemoryCopyResultDeviceReset:
      return CL_DEVICE_NOT_AVAILABLE;
    default:
      return CL_OUT_OF_RESOURCES;
  }
}

cl_int formosa_memory_submit_copy(MemoryDomain src_domain, uint64_t src_addr,
                                  MemoryDomain dst_domain, uint64_t dst_addr,
                                  size_t size,
                                  FsaMemoryCopyCompletion *completion) {
  if (completion == nullptr) return CL_INVALID_VALUE;
  *completion = (FsaMemoryCopyCompletion){};
  if (size == 0) return CL_SUCCESS;
  if (fsa_cmd_memory_copy(src_domain, src_addr, dst_domain, dst_addr, size,
                          completion) != 0) {
    return CL_OUT_OF_RESOURCES;
  }
  return CL_SUCCESS;
}

cl_int formosa_memory_copy(MemoryDomain src_domain, uint64_t src_addr,
                           MemoryDomain dst_domain, uint64_t dst_addr,
                           size_t size) {
  FsaMemoryCopyCompletion completion = {};
  cl_int submit_status = formosa_memory_submit_copy(
      src_domain, src_addr, dst_domain, dst_addr, size, &completion);
  if (submit_status != CL_SUCCESS) return submit_status;
  if (size == 0) return CL_SUCCESS;

  MemoryCopyResult result = kMemoryCopyResultPending;
  const int wait_status = fsa_wait_for_memory_copy(completion, 0, &result);
  if (wait_status == 0) return CL_SUCCESS;

  POCL_MSG_ERR("Formosa memory copy failed with result %d\n", (int)result);
  return formosa_memory_copy_result_to_cl(result);
}
