#include "pocl-formosa-util.h"

#include <libcomm/comm.h>
#include <libcomm/msg.h>

#include <cstdio>
#include <fstream>
#include <iostream>

#include "casvp-config/casvp-config.h"
#include "falloc/fsa_mem_allocator.h"
#include "pocl_cl.h"

int fsa_check_occupancy(uint32_t group_size, uint32_t *max_local_mem) {
  // check group size
  uint64_t warps_per_core, threads_per_warp;
  uint32_t threads_per_core = warps_per_core * threads_per_warp;
  if (group_size > threads_per_core) {
    printf(
        "Error: cannot schedule kernel with group_size > threads_per_core "
        "(%d,%d)\n",
        group_size, threads_per_core);
    return -1;
  }

  // calculate groups occupancy
  int warps_per_group = (group_size + threads_per_warp - 1) / threads_per_warp;
  int groups_per_core = warps_per_core / warps_per_group;

  // check local memory capacity
  if (max_local_mem) {
    uint64_t local_mem_size;
    *max_local_mem = local_mem_size / groups_per_core;
  }

  return 0;
}

int fsa_copy_to_dev(formosa_buffer_data_t *buffer_data, const void *host_ptr,
                    uint64_t dst_offset, size_t size) {
  if (buffer_data == nullptr || host_ptr == nullptr) return -1;
  if ((dst_offset + size) > buffer_data->buf_size) return -1;
  msg_t *msg =
      msg_create(0, WRITE, size, buffer_data->buf_address + dst_offset);
  if (msg == nullptr) return -1;
  msg->payload = static_cast<uint8_t *>(const_cast<void *>(host_ptr));
  int status = ipc_send_write_msg(buffer_data->client_fd, msg);
  msg_destroy(msg);
  return status;
}

int fsa_copy_from_dev(formosa_buffer_data_t *buffer_data, void *host_ptr,
                      uint64_t src_offset, size_t size) {
  if (buffer_data == nullptr || host_ptr == nullptr) return -1;
  if ((src_offset + size) > buffer_data->buf_size) return -1;
  msg_t *msg = msg_create(0, READ, size, buffer_data->buf_address + src_offset);
  if (msg == nullptr) return -1;
  msg->payload = static_cast<uint8_t *>(host_ptr);
  int status = ipc_send_read_msg(buffer_data->client_fd, msg);
  msg_destroy(msg);
  return status;
}

int fsa_upload_kernel_file(const char *filename, pocl_formosa_data_t *dd) {
  if (filename == nullptr || dd == nullptr) return -1;
  std::ifstream file(filename);
  if (!file) {
    std::cerr << "Error: failed to open kernel file " << filename << std::endl;
    return -1;
  }
  file.seekg(0, file.end);
  uint64_t size = file.tellg();
  file.seekg(0, file.beg);
  char *data = new char[size];
  if (data == NULL) {
    std::cerr << "Error: failed to allocate memory for kernel" << std::endl;
    file.close();
    return -1;
  }
  file.read(data, size);
  file.close();

  int status = 0;
  dd->kernel_buffer = new formosa_buffer_data_t;
  if (dd->kernel_buffer == nullptr) {
    status = -1;
    goto UPLOAD_KERNEL_FILE_ERROR;
  }

  void *addr;
  status = fsaMalloc(&addr, size);
  if (status != 0) {
    goto UPLOAD_KERNEL_FILE_ERROR;
  }

  dd->kernel_buffer->client_fd = dd->client_fd;
  dd->kernel_buffer->buf_size = size;
  dd->kernel_buffer->buf_address = (uint64_t)addr;

  status = fsa_copy_to_dev(dd->kernel_buffer, data, 0, size);
  if (status != 0) {
    fsaFree(static_cast<void *>(addr));
  UPLOAD_KERNEL_FILE_ERROR:
    POCL_MEM_FREE(dd->kernel_buffer);
  }
  delete[] data;
  return status;
}

int fsa_wait_ack(pocl_formosa_data_t *dd) {
  if (dd == nullptr) return -1;
  uint64_t ack;  // acknowledge from device
  int status;
  do {
    status = fsa_read_csr(dd, CASVP_FORMOSA_CSR_ACK, &ack);
    if (status != 0) return -1;
    nanosleep((const struct timespec[]){{0, 10000000}}, NULL);  // 10ms
  } while (ack == 0);
  return 0;
}

int fsa_write_csr(pocl_formosa_data_t *dd, uint64_t addr, uint64_t value) {
  if (dd == nullptr) return -1;
  msg_t *msg = msg_create(0, WRITE, 8, addr);
  if (msg == nullptr) return -1;
  msg->payload = reinterpret_cast<uint8_t *>(&value);
  int status = ipc_send_write_msg(dd->client_fd, msg);
  msg_destroy(msg);
  return status;
}

int fsa_read_csr(pocl_formosa_data_t *dd, uint64_t addr, uint64_t *value) {
  if (dd == nullptr) return -1;
  msg_t *msg = msg_create(0, READ, 8, addr);
  if (msg == nullptr) return -1;
  msg->payload = reinterpret_cast<uint8_t *>(value);
  int status = ipc_send_read_msg(dd->client_fd, msg);
  msg_destroy(msg);
  return status;
}
