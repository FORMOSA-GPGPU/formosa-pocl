#include "formosa-memory.h"

#include "formosa-hal/hal.h"
#include "formosa-util.h"
#include "pocl-formosa-internal.h"
#include "pocl_debug.h"

namespace {

struct MemoryCopySubmitArgs {
  FsaMemoryCopyInfo info;
};

FsaCommandSubmitStatus submit_memory_copy(void *context,
                                          FsaCompletionToken *token) {
  const MemoryCopySubmitArgs *args =
      static_cast<const MemoryCopySubmitArgs *>(context);
  return fsa_cmd_memory_copy(&args->info, token);
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
  MemoryCopySubmitArgs args{};
  args.info.struct_size = sizeof(args.info);
  args.info.source.domain = src_domain;
  args.info.source.range.address = src_addr;
  args.info.source.range.size = size;
  args.info.destination.domain = dst_domain;
  args.info.destination.range.address = dst_addr;
  args.info.destination.range.size = size;
  const FsaCommandSubmitStatus submit_status =
      pocl_fsa_submit_with_backpressure(submit_memory_copy, (void *)&args,
                                        token);

  switch (submit_status) {
    case kFsaCommandSubmitAccepted:
      return CL_SUCCESS;
    case kFsaCommandSubmitInvalidArgument:
      return CL_INVALID_VALUE;
    case kFsaCommandSubmitTransportError:
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

cl_int formosa_memory_copy_rows(pocl_mem_identifier *dst_mem, void *dst_host,
                                size_t dst_base, size_t dst_row_pitch,
                                size_t dst_slice_pitch,
                                pocl_mem_identifier *src_mem,
                                const void *src_host, size_t src_base,
                                size_t src_row_pitch, size_t src_slice_pitch,
                                const size_t *region, size_t row_bytes) {
  if ((dst_mem == nullptr && dst_host == nullptr) ||
      (src_mem == nullptr && src_host == nullptr))
    return CL_INVALID_VALUE;
  for (size_t z = 0; z < region[2]; ++z) {
    for (size_t y = 0; y < region[1]; ++y) {
      size_t dst_offset = dst_base + z * dst_slice_pitch + y * dst_row_pitch;
      size_t src_offset = src_base + z * src_slice_pitch + y * src_row_pitch;
      uint64_t dst_addr, src_addr;
      cl_int err;
      if (dst_mem != nullptr) {
        err = formosa_memory_resolve_buffer_address(dst_mem, dst_offset,
                                                    row_bytes, &dst_addr);
        if (err != CL_SUCCESS) return err;
      } else {
        dst_addr = reinterpret_cast<uintptr_t>(static_cast<char *>(dst_host) +
                                               dst_offset);
      }
      if (src_mem != nullptr) {
        err = formosa_memory_resolve_buffer_address(src_mem, src_offset,
                                                    row_bytes, &src_addr);
        if (err != CL_SUCCESS) return err;
      } else {
        src_addr = reinterpret_cast<uintptr_t>(
            static_cast<const char *>(src_host) + src_offset);
      }
      err = formosa_memory_copy(
          src_mem ? kMemoryDomainDevice : kMemoryDomainHost, src_addr,
          dst_mem ? kMemoryDomainDevice : kMemoryDomainHost, dst_addr,
          row_bytes);
      if (err != CL_SUCCESS) return err;
    }
  }
  return CL_SUCCESS;
}

static size_t rect_base(const size_t *origin, size_t row_pitch,
                        size_t slice_pitch) {
  return origin[0] + origin[1] * row_pitch + origin[2] * slice_pitch;
}

cl_int formosa_memory_exec_rect_command(_cl_command_node *node) {
  _cl_command_t *cmd = &node->command;
  unsigned mem_id = node->device->global_mem_id;
  switch (node->type) {
    case CL_COMMAND_READ_BUFFER_RECT: {
      _cl_command_read_rect *r = &cmd->read_rect;
      return formosa_memory_copy_rows(
          nullptr, r->dst_host_ptr,
          rect_base(r->host_origin, r->host_row_pitch, r->host_slice_pitch),
          r->host_row_pitch, r->host_slice_pitch, &r->src->device_ptrs[mem_id],
          nullptr,
          rect_base(r->buffer_origin, r->buffer_row_pitch,
                    r->buffer_slice_pitch),
          r->buffer_row_pitch, r->buffer_slice_pitch, r->region, r->region[0]);
    }
    case CL_COMMAND_WRITE_BUFFER_RECT: {
      _cl_command_write_rect *r = &cmd->write_rect;
      return formosa_memory_copy_rows(
          &r->dst->device_ptrs[mem_id], nullptr,
          rect_base(r->buffer_origin, r->buffer_row_pitch,
                    r->buffer_slice_pitch),
          r->buffer_row_pitch, r->buffer_slice_pitch, nullptr, r->src_host_ptr,
          rect_base(r->host_origin, r->host_row_pitch, r->host_slice_pitch),
          r->host_row_pitch, r->host_slice_pitch, r->region, r->region[0]);
    }
    case CL_COMMAND_COPY_BUFFER_RECT: {
      _cl_command_copy_rect *r = &cmd->copy_rect;
      return formosa_memory_copy_rows(
          &r->dst->device_ptrs[mem_id], nullptr,
          rect_base(r->dst_origin, r->dst_row_pitch, r->dst_slice_pitch),
          r->dst_row_pitch, r->dst_slice_pitch, &r->src->device_ptrs[mem_id],
          nullptr,
          rect_base(r->src_origin, r->src_row_pitch, r->src_slice_pitch),
          r->src_row_pitch, r->src_slice_pitch, r->region, r->region[0]);
    }
    default:
      return CL_INVALID_OPERATION;
  }
}
