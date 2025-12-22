#include "formosa-util.h"

#include <linux/elf.h>
#include <unistd.h>

#include <iostream>
#include <sstream>
#include <string>
#include <tuple>
#include <vector>

#include "formosa-hal/formosa-hal.h"
#include "formosa-llvm-util.h"
#include "pocl.h"
#include "pocl_cache.h"
#include "pocl_debug.h"
#include "pocl_file_util.h"
#include "pocl_runtime_config.h"
#include "pocl_util.h"

namespace {
void deserialize_kernel_status(const uint8_t *raw, KernelStatus *status) {
  uint64_t code = *reinterpret_cast<const uint64_t *>(raw);
  switch (code) {
    case 0:
      status->code = kKernelOkay;
      break;
    case 1:
      status->code = kKernelBadDimension;
      break;
    case 2:
      status->code = kKernelException;
      break;
    default:
      status->code = kKernelUnknownError;
      break;
  }
  status->mcause = *reinterpret_cast<const uint64_t *>(raw + 8);
  status->mepc = *reinterpret_cast<const uint64_t *>(raw + 16);
  status->mtval = *reinterpret_cast<const uint64_t *>(raw + 24);
}
}  // namespace

int pocl_fsa_check_occupancy(uint32_t group_size, uint64_t *max_local_mem) {
  // check group size
  uint32_t warps_per_core = fsa_warps_per_core();
  uint32_t threads_per_warp = fsa_threads_per_warp();
  uint32_t threads_per_core = warps_per_core * threads_per_warp;
  if (group_size > threads_per_core) {
    POCL_MSG_ERR(
        "Cannot schedule kernel with group_size (%d) > threads_per_core (%d)\n",
        group_size, threads_per_core);
    return -1;
  }

  // calculate groups occupancy
  int warps_per_group = (group_size + threads_per_warp - 1) / threads_per_warp;
  int groups_per_core = warps_per_core / warps_per_group;

  // check local memory capacity
  if (max_local_mem) {
    uint64_t local_mem_size = fsa_local_mem_size();
    *max_local_mem = local_mem_size / groups_per_core;
  }

  return 0;
}

int pocl_fsa_get_elf_name(cl_program program, cl_uint device_i,
                          char *elf_name) {
  if (program == nullptr || elf_name == nullptr) return -1;

  char program_bc[POCL_MAX_PATHNAME_LENGTH];
  pocl_cache_program_bc_path(program_bc, program, device_i);

  // remove extension name
  char *last_dot = strrchr(program_bc, '.');
  if (last_dot != nullptr) *last_dot = '\0';

  strcpy(elf_name, program_bc);
  strncat(elf_name, ".fsa.bin", POCL_MAX_PATHNAME_LENGTH - 1);
  return 0;
}

int pocl_fsa_upload_kernel(const char *elf_file, pocl_formosa_data_t *dd,
                           uint64_t *kernel_dev_addr) {
  if (elf_file == nullptr || dd == nullptr) return -1;
  uint64_t kernel_size = 0;
  FILE *elf = fopen(elf_file, "rb");
  rewind(elf);
  Elf64_Ehdr ehdr;
  [[maybe_unused]] size_t read_size;
  read_size = fread(&ehdr, sizeof(ehdr), 1, elf);
  // 1st pass, iterate all program headers to find the minimum base address
  for (int i = 0; i < ehdr.e_phnum; i++) {
    fseek(elf, ehdr.e_phoff + i * (ehdr.e_phentsize), SEEK_SET);
    Elf64_Phdr phdr;
    // read prog header from elf
    read_size = fread(&phdr, sizeof(phdr), 1, elf);
    if (phdr.p_type != PT_LOAD) continue;
    uint64_t addr = phdr.p_paddr;
    if (addr + phdr.p_memsz > kernel_size) kernel_size = addr + phdr.p_memsz;
  }
  void *kernel_start_addr;
  if (fsa_malloc(&kernel_start_addr, kernel_size)) {
    POCL_MSG_ERR(
        "Failed to allocate FSA device side memory in fsa_upload_kernel");
    fclose(elf);
    return -1;
  }

  *kernel_dev_addr = (uint64_t)kernel_start_addr;
  // move file pointer to beginning of file
  rewind(elf);
  uint8_t *host_ptr = (uint8_t *)calloc(1, sizeof(uint8_t) * kernel_size);
  uint64_t offset = 0;

  // 2nd pass, copy all program headers to device memory
  for (int i = 0; i < ehdr.e_phnum; i++) {
    fseek(elf, ehdr.e_phoff + i * (ehdr.e_phentsize), SEEK_SET);
    Elf64_Phdr phdr;
    read_size = fread(&phdr, sizeof(phdr), 1, elf);
    if (phdr.p_type != PT_LOAD) continue;
    uint64_t size = phdr.p_filesz;
    uint64_t offset = phdr.p_paddr;  // ORIGIN in link.ld is zero
    fseek(elf, phdr.p_offset, SEEK_SET);
    if (size) {
      read_size = fread(host_ptr + offset, size, 1, elf);
    }
    if (size < phdr.p_memsz) size = phdr.p_memsz;  // trail-zero-filling

    POCL_MSG_PRINT_INFO("Copy %lx to %lx with size %lx\n",
                        (uint64_t)host_ptr + offset,
                        (uint64_t)kernel_start_addr + offset, size);
    fsa_copy_to_dev((uint64_t)kernel_start_addr + offset, host_ptr + offset,
                    size);
  }
  free(host_ptr);
  fclose(elf);
  return 0;
}

int pocl_fsa_wait_ack(pocl_formosa_data_t *dd, uintptr_t completion_signal, uintptr_t device_kernel_status_addr) {
  if (dd == nullptr || completion_signal == 0) return -1;
  // polling the completion_signal until it is set to non-zero value
  fsa_wait_for_completion(
      completion_signal,
      0);  // blocking wait

  uint8_t status_raw[sizeof(KernelStatus)];
  int err = fsa_copy_from_dev(device_kernel_status_addr, status_raw, sizeof(status_raw));
  if (err != 0) {
    POCL_MSG_ERR("Failed to read kernel status from device (%d)\n", err);
    return -1;
  }

  KernelStatus status;
  deserialize_kernel_status(status_raw, &status);

  switch (status.code) {
    case kKernelOkay:
      break;
    case kKernelBadDimension:
      POCL_MSG_ERR("Bad dimension\n");
      break;
    case kKernelException:
      POCL_MSG_ERR(
          "\nException occurs:\n"
          "\tmcause: 0x%08lx\n"
          "\tmepc:   0x%08lx\n"
          "\tmtval:  0x%08lx\n",
          status.mcause, status.mepc, status.mtval);
      break;
    default:
      __attribute__((fallthrough));
    case kKernelUnknownError:
      POCL_MSG_ERR(
          "Unknown error occurred in kernel execution\n"
          "\tmcause: 0x%08lx\n"
          "\tmepc:   0x%08lx\n"
          "\tmtval:  0x%08lx\n",
          status.mcause, status.mepc, status.mtval);
  }

  return (status.code == kKernelOkay) ? 0 : -1;
}

static int exec(const char *cmd, std::ostream &out) {
  char buffer[128];
  auto pipe = popen(cmd, "r");
  if (!pipe) {
    return -1;
  }
  while (!feof(pipe)) {
    if (fgets(buffer, 128, pipe) != nullptr) out << buffer;
  }
  return pclose(pipe);
}

std::stringstream generate_command(std::vector<std::string> &args) {
  std::stringstream ss_cmd;
  for (const auto &arg : args) {
    ss_cmd << arg << " ";
  }
  return ss_cmd;
}

std::tuple<int, std::stringstream> compile_source(char *src_path,
                                                  const char *obj_path,
                                                  std::string clang_path,
                                                  std::string build_cflags) {
  int err;
  std::stringstream ss_cmd, ss_out;
  std::vector<std::string> args = {clang_path, build_cflags, src_path,
                                   "-c",       "-o",         obj_path};
  ss_cmd = generate_command(args);
  POCL_MSG_PRINT_LLVM("Running \"%s\"\n", ss_cmd.str().c_str());
  err = exec(ss_cmd.str().c_str(), ss_out);
  return std::make_tuple(err, std::move(ss_out));
}

int pocl_fsa_compile_program(char **kernel_names, int *num_kernels,
                             char *str_program_fsa_bin, char *compiler_options,
                             void *llvm_module) {
  int err;
  std::string llvm_path = FORMOSA_LLVM;
  std::string llvm_objdump_path = llvm_path + "/bin/llvm-objdump";
  std::string build_cflags = "-mcpu=formosa-gpgpu ";
  std::string extra_cflags = pocl_get_string_option("POCL_FORMOSA_CFLAGS", "");
  if (compiler_options != nullptr) {
    build_cflags += std::string(compiler_options) + " ";
  }
  if (extra_cflags == "") {
    POCL_MSG_WARN(
        "Environment variable 'POCL_FORMOSA_CFLAGS' is not set, default to "
        "-O1 -mllvm -fsa-pdom-level\n");
    build_cflags += "-O1 -mllvm -fsa-pdom-level";
  } else {
    if (extra_cflags.find("-fsa-ics-first") != std::string::npos)
      build_cflags += " -Xclang -fsa-ics-first-cg ";
    build_cflags += extra_cflags;
  }

  std::string build_ldflags = "-lm ";
  std::string extra_ldflags =
      pocl_get_string_option("POCL_FORMOSA_LDFLAGS", "");
  if (extra_ldflags == "") {
    POCL_MSG_WARN(
        "Environment variable 'POCL_FORMOSA_LDFLAGS' is not set, default to "
        "-fuse-ld=lld -nostartfiles\n");
    build_ldflags += "-fuse-ld=lld -nostartfiles";
  } else {
    build_ldflags += extra_ldflags;
  }

  char bitcode_path[POCL_MAX_PATHNAME_LENGTH];
  err = pocl_mk_tempname(bitcode_path, "/tmp/pocl_formosa_program", ".bc",
                         nullptr);
  if (err != 0) return err;

  char elf_path[POCL_MAX_PATHNAME_LENGTH];
  memcpy(elf_path, str_program_fsa_bin, strlen(str_program_fsa_bin) + 1);

  pocl_fsa_build_kernel(llvm_module, bitcode_path, (unsigned *)num_kernels,
                        kernel_names);

  std::string clang_path(CLANGCC);
  if (!llvm_path.empty()) {
    clang_path = llvm_path + "/bin/clang";
  }

  char kernel_util_path[POCL_MAX_PATHNAME_LENGTH];
  char start_file_path[POCL_MAX_PATHNAME_LENGTH];
  char linker_script_path[POCL_MAX_PATHNAME_LENGTH];
  char printf_src_path[POCL_MAX_PATHNAME_LENGTH];
  char putchar_src_path[POCL_MAX_PATHNAME_LENGTH];
  pocl_get_srcdir_or_datadir(kernel_util_path, "/lib/CL/devices", "",
                             "/formosa/kernel_util.cl");
  pocl_get_srcdir_or_datadir(start_file_path, "/lib/CL/devices", "",
                             "/formosa/start.S");
  pocl_get_srcdir_or_datadir(linker_script_path, "/lib/CL/devices", "",
                             "/formosa/link.ld");
  pocl_get_srcdir_or_datadir(printf_src_path, "/lib/CL/devices", "",
                             "/formosa/printf.c");
  pocl_get_srcdir_or_datadir(putchar_src_path, "/lib/CL/devices", "",
                             "/formosa/putchar.c");

  const char *printf_obj_path = "/tmp/printf.o";
  const char *putchar_obj_path = "/tmp/putchar.o";
  const char *printf_lib_path = "/tmp/libprintf.a";

  // First compile printf.c and putchar.c
  std::stringstream ss_out;
  std::tie(err, ss_out) = compile_source(printf_src_path, printf_obj_path,
                                         clang_path, build_cflags + " -fPIC ");
  if (err != 0) {
    POCL_MSG_ERR("%s\n", ss_out.str().c_str());
    return err;
  }
  std::tie(err, ss_out) = compile_source(putchar_src_path, putchar_obj_path,
                                         clang_path, build_cflags + " -fPIC ");
  if (err != 0) {
    POCL_MSG_ERR("%s\n", ss_out.str().c_str());
    return err;
  }

  // Archive printf.o and putchar.o into libprintf.a
  std::stringstream ss_cmd;
  std::string archive_path = llvm_path + "/bin/llvm-ar";
  std::vector<std::string> args = {archive_path, " -rcs ", printf_lib_path,
                                   printf_obj_path, putchar_obj_path};
  ss_cmd = generate_command(args);
  POCL_MSG_PRINT_LLVM("Running \"%s\"\n", ss_cmd.str().c_str());
  err = exec(ss_cmd.str().c_str(), ss_out);
  if (err != 0) {
    POCL_MSG_ERR("%s\n", ss_out.str().c_str());
    return err;
  }

  // Link kernel program with predefined kernel functions and libprintf.a
  args = {clang_path,       build_cflags + " -fPIE ",
          start_file_path,  bitcode_path,
          " -L/tmp ",       " -lprintf ",
          kernel_util_path, build_ldflags,
          " -T ",           linker_script_path,
          " -Wl,-pie -o ",  elf_path};
  ss_cmd = generate_command(args);
  POCL_MSG_PRINT_LLVM("Running \"%s\"\n", ss_cmd.str().c_str());
  err = exec(ss_cmd.str().c_str(), ss_out);
  if (err != 0) {
    POCL_MSG_ERR("%s\n", ss_out.str().c_str());
    return err;
  }

  if (POCL_DEBUGGING_ON) {
    std::string objdump_path(llvm_objdump_path);

    std::stringstream ss_cmd, ss_out;
    std::vector<std::string> args = {objdump_path, "-d", elf_path, ">",
                                     "program.dump"};
    ss_cmd = generate_command(args);
    POCL_MSG_PRINT_LLVM("Running \"%s\"\n", ss_cmd.str().c_str());
    err = exec(ss_cmd.str().c_str(), ss_out);
    if (err != 0) {
      POCL_MSG_ERR("%s\n", ss_out.str().c_str());
      return err;
    }
  }
  return 0;
}
