#include "formosa-memory.h"

#include "formosa-util.h"
#include "pocl-formosa-internal.h"
#include "pocl_debug.h"

namespace {

struct MemoryCopySubmitArgs {
  MemoryDomain src_domain;
  uint64_t src_addr;
  MemoryDomain dst_domain;
  uint64_t dst_addr;
  size_t size;
};

FsaCompletionSubmitStatus submit_memory_copy(void *context,
                                             FsaCompletionToken *token) {
  const MemoryCopySubmitArgs *args =
      static_cast<const MemoryCopySubmitArgs *>(context);
  return fsa_cmd_memory_copy(args->src_domain, args->src_addr, args->dst_domain,
                             args->dst_addr, args->size, token);
}

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

cl_int formosa_memory_copy_outcome_to_cl(FsaCompletionResult outcome) {
  if (outcome == FSA_COMPLETION_RESULT_SUCCESS) return CL_SUCCESS;
  if (outcome == FSA_COMPLETION_RESULT_FIRMWARE_REBOOT)
    return CL_DEVICE_NOT_AVAILABLE;

  switch (outcome) {
    case kMemoryCopyStatusOverlap:
      return CL_MEM_COPY_OVERLAP;
    case kMemoryCopyStatusInvalidAddress:
    case kMemoryCopyStatusInvalidRange:
    case kMemoryCopyStatusInvalidDomainPair:
      return CL_INVALID_VALUE;
    default:
      return CL_OUT_OF_RESOURCES;
  }
}

cl_int formosa_memory_submit_copy(MemoryDomain src_domain, uint64_t src_addr,
                                  MemoryDomain dst_domain, uint64_t dst_addr,
                                  size_t size, FsaCompletionToken *token) {
  if (token == nullptr) return CL_INVALID_VALUE;
  *token = 0;
  if (size == 0) return CL_SUCCESS;
  if (!fsa_hal_is_available()) {
    formosa_mark_unavailable();
    return CL_DEVICE_NOT_AVAILABLE;
  }
  const MemoryCopySubmitArgs args = {src_domain, src_addr, dst_domain, dst_addr,
                                     size};
  const FsaCompletionSubmitStatus submit_status =
      pocl_fsa_submit_with_backpressure(submit_memory_copy, (void *)&args,
                                        token);

  switch (submit_status) {
    case kFsaCompletionSubmitAccepted:
      return CL_SUCCESS;
    case kFsaCompletionSubmitInvalidArgument:
      return CL_INVALID_VALUE;
    case kFsaCompletionSubmitTransportError:
      if (!fsa_hal_is_available()) {
        /* Token may remain owned after ambiguous wr_ptr publish; session
         * fail-stop reclaims via reset/uninit rather than caller release. */
        formosa_mark_unavailable();
        return CL_DEVICE_NOT_AVAILABLE;
      }
      /* A pre-doorbell transport failure rolled the token back and does not
       * invalidate the device session. */
      return CL_OUT_OF_RESOURCES;
  }
  return CL_OUT_OF_RESOURCES;
}

cl_int formosa_memory_copy(MemoryDomain src_domain, uint64_t src_addr,
                           MemoryDomain dst_domain, uint64_t dst_addr,
                           size_t size) {
  FsaCompletionToken token = 0;
  cl_int submit_status = formosa_memory_submit_copy(
      src_domain, src_addr, dst_domain, dst_addr, size, &token);
  if (submit_status != CL_SUCCESS) return submit_status;
  if (size == 0) return CL_SUCCESS;

  FsaCompletionResult result = FSA_COMPLETION_RESULT_PENDING;
  const cl_int wait_status = pocl_fsa_wait_completion_result(token, &result);
  if (wait_status == CL_DEVICE_NOT_AVAILABLE) {
    return CL_DEVICE_NOT_AVAILABLE;
  }

  if (wait_status != CL_SUCCESS) return CL_OUT_OF_RESOURCES;

  if (result == FSA_COMPLETION_RESULT_SUCCESS) return CL_SUCCESS;

  POCL_MSG_ERR("Formosa memory copy failed (wait=%d, outcome=%d)\n",
               wait_status, (int)result);
  return formosa_memory_copy_outcome_to_cl(result);
}
