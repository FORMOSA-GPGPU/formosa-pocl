#include "formosa-util.h"

#include <inttypes.h>
#include <linux/elf.h>
#include <unistd.h>

#include <iostream>
#include <sstream>
#include <string>
#include <tuple>
#include <vector>

#include "formosa-hal/formosa-hal.h"
#include "formosa-llvm-util.h"
#include "formosa-real/formosa-config.h"
#include "pocl.h"
#include "pocl_cache.h"
#include "pocl_debug.h"
#include "pocl_file_util.h"
#include "pocl_runtime_config.h"
#include "pocl_util.h"

#ifndef R_RISCV_NONE
#define R_RISCV_NONE 0
#endif

#ifndef R_RISCV_RELATIVE
#define R_RISCV_RELATIVE 3
#endif

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

int apply_kernel_relocations(FILE *elf, const Elf64_Ehdr &ehdr,
                             uint8_t *host_image, uint64_t image_size,
                             uint64_t image_base) {
  if (ehdr.e_shoff == 0 || ehdr.e_shnum == 0) return 0;

  for (int i = 0; i < ehdr.e_shnum; ++i) {
    if (fseek(elf, ehdr.e_shoff + i * ehdr.e_shentsize, SEEK_SET) != 0)
      return -1;

    Elf64_Shdr shdr;
    if (fread(&shdr, sizeof(shdr), 1, elf) != 1) return -1;

    if (shdr.sh_type == SHT_RELA) {
      if (shdr.sh_entsize != 0 && shdr.sh_entsize != sizeof(Elf64_Rela)) {
        POCL_MSG_ERR("Unexpected RELA entry size %lu in kernel ELF\n",
                     (unsigned long)shdr.sh_entsize);
        return -1;
      }

      const size_t num_entries = shdr.sh_size / sizeof(Elf64_Rela);
      if (fseek(elf, shdr.sh_offset, SEEK_SET) != 0) return -1;

      for (size_t j = 0; j < num_entries; ++j) {
        Elf64_Rela rela;
        if (fread(&rela, sizeof(rela), 1, elf) != 1) return -1;

        const uint32_t type = ELF64_R_TYPE(rela.r_info);
        switch (type) {
          case R_RISCV_RELATIVE: {
            if (rela.r_offset + sizeof(uint64_t) > image_size) {
              POCL_MSG_ERR("Kernel relocation target 0x%lx is out of bounds\n",
                           (unsigned long)rela.r_offset);
              return -1;
            }
            const uint64_t relocated = image_base + rela.r_addend;
            memcpy(host_image + rela.r_offset, &relocated, sizeof(relocated));
            break;
          }
          case R_RISCV_NONE:
            break;
          default:
            POCL_MSG_ERR("Unsupported kernel relocation type %u in kernel ELF\n",
                         type);
            return -1;
        }
      }
    } else if (shdr.sh_type == SHT_REL) {
      POCL_MSG_ERR("REL relocations are not supported in kernel ELF\n");
      return -1;
    }
  }

  return 0;
}
}  // namespace

int pocl_fsa_check_occupancy(uint32_t group_size, uint64_t local_mem_per_group,
                             uint64_t *max_local_mem) {
  // Hardware limits
  uint32_t warps_per_core = fsa_warps_per_core();
  uint32_t threads_per_warp = fsa_threads_per_warp();
  uint32_t threads_per_core = warps_per_core * threads_per_warp;
  uint64_t local_mem_size = fsa_local_mem_size();

  // Sanity check
  if (group_size == 0) {
    POCL_MSG_ERR("group_size must be > 0\n");
    return -1;
  }

  // Check thread capacity
  if (group_size > threads_per_core) {
    POCL_MSG_ERR(
        "Cannot schedule kernel: group_size (%u) > threads_per_core (%u)\n",
        group_size, threads_per_core);
    return -1;
  }

  // --- Thread-based occupancy ---
  // ceil(group_size / threads_per_warp)
  uint32_t warps_per_group =
      (group_size + threads_per_warp - 1) / threads_per_warp;

  // max groups limited by warp slots
  uint32_t groups_by_threads = warps_per_core / warps_per_group;

  if (groups_by_threads == 0) {
    POCL_MSG_ERR("No available slots due to warp/thread constraint\n");
    return -1;
  }

  // --- Local memory-based occupancy ---
  uint32_t groups_by_local_mem;

  if (local_mem_per_group == 0) {
    // no local memory usage, not a limiting factor
    groups_by_local_mem = groups_by_threads;
  } else {
    if (local_mem_per_group > local_mem_size) {
      POCL_MSG_ERR("Cannot schedule kernel: local_mem_per_group (%" PRIu64
                   ") > local_mem_size (%" PRIu64 ")\n",
                   local_mem_per_group, local_mem_size);
      return -1;
    }

    groups_by_local_mem = local_mem_size / local_mem_per_group;
  }

  // --- Final occupancy (take the bottleneck) ---
  if (std::min(groups_by_threads, groups_by_local_mem) == 0) {
    POCL_MSG_ERR("Kernel cannot have any resident workgroups per core\n");
    return -1;
  }

  // Output: max local memory per WG to preserve thread-based occupancy
  if (max_local_mem) {
    // evenly distribute local memory across thread-limited WG slots
    *max_local_mem = local_mem_size / groups_by_threads;
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
  if (host_ptr == nullptr) {
    fclose(elf);
    return -1;
  }

  // 2nd pass, load all PT_LOAD segments into the host-side image.
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
  }

  if (apply_kernel_relocations(elf, ehdr, host_ptr, kernel_size,
                               *kernel_dev_addr) != 0) {
    free(host_ptr);
    fclose(elf);
    return -1;
  }

  // 3rd pass, copy the relocated loadable image to device memory.
  for (int i = 0; i < ehdr.e_phnum; i++) {
    fseek(elf, ehdr.e_phoff + i * (ehdr.e_phentsize), SEEK_SET);
    Elf64_Phdr phdr;
    read_size = fread(&phdr, sizeof(phdr), 1, elf);
    if (phdr.p_type != PT_LOAD) continue;

    uint64_t size = phdr.p_memsz;
    uint64_t offset = phdr.p_paddr;  // ORIGIN in link.ld is zero

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

int pocl_fsa_wait_ack(pocl_formosa_data_t *dd, uintptr_t completion_signal,
                      uintptr_t device_kernel_status_addr) {
  if (dd == nullptr || completion_signal == 0) return -1;
  // polling the completion_signal until it is set to non-zero value
  fsa_wait_for_completion(completion_signal,
                          0);  // blocking wait

  uint8_t status_raw[sizeof(KernelStatus)];
  int err = fsa_copy_from_dev(device_kernel_status_addr, status_raw,
                              sizeof(status_raw));
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
        "-O2\n");
    build_cflags += "-O2";
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
  pocl_get_srcdir_or_datadir(kernel_util_path, "/lib/kernel", "",
                             "/formosa/kernel_util.cl");
  pocl_get_srcdir_or_datadir(start_file_path, "/lib/kernel", "",
                             "/formosa/start.S");
  pocl_get_srcdir_or_datadir(linker_script_path, "/lib/kernel", "",
                             "/formosa/link.ld");
  std::stringstream ss_out;

  // Link kernel program with predefined kernel functions and libprintf.a
  std::stringstream ss_cmd;
  std::vector<std::string> args = {clang_path,       build_cflags + " -fPIE ",
                                   start_file_path,  bitcode_path,
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
