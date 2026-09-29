#include "simtsim-util.h"

#include <sstream>
#include <string>
#include <vector>

#include "simtsim-llvm-util.h"
#include "pocl.h"
#include "pocl_cache.h"
#include "pocl_file_util.h"
#include "pocl_util.h"

static int exec(const char *cmd, std::ostream &out) {
  char buffer[128];
  auto pipe = popen(cmd, "r");
  if (!pipe) {
    return -1;
  }
  while (!feof(pipe)) {
    if (fgets(buffer, 128, pipe) != nullptr)
      out << buffer;
  }
  return pclose(pipe);
}

static std::stringstream generate_command(std::vector<std::string> &args) {
  std::stringstream ss_cmd;
  for (const auto &arg : args) {
    ss_cmd << arg << " ";
  }
  return ss_cmd;
}

int pocl_simtsim_compile_program(char **kernel_names, int *num_kernels,
                                 char *str_program_fsa_bin,
                                 char *compiler_options, void *llvm_module) {
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
        "-O3\n");
    build_cflags += "-O3 -finline-functions -mllvm -inline-threshold=10000";
  } else {
    build_cflags += extra_cflags;
  }

  std::string build_ldflags = "-lm ";
  std::string extra_ldflags =
      pocl_get_string_option("POCL_FORMOSA_LDFLAGS", "");
  if (extra_ldflags == "") {
    POCL_MSG_WARN(
        "Environment variable 'POCL_FORMOSA_LDFLAGS' is not set, default to "
        "-fuse-ld=lld -nostartfiles\n");
    build_ldflags += "-fuse-ld=lld -nostartfiles -nodefaultlibs";
  } else {
    build_ldflags += extra_ldflags;
  }

  char bitcode_path[POCL_MAX_PATHNAME_LENGTH];
  err = pocl_mk_tempname(bitcode_path, "/tmp/pocl_simtsim_program", ".bc",
                         nullptr);
  if (err != 0)
    return err;

  char elf_path[POCL_MAX_PATHNAME_LENGTH];
  memcpy(elf_path, str_program_fsa_bin, strlen(str_program_fsa_bin) + 1);

  pocl_fsa_build_kernel(llvm_module, bitcode_path, (unsigned *)num_kernels,
                        kernel_names);

  const char *default_clang = CLANGCC;
#ifdef FORMOSA_CLANG_PATH
  default_clang = FORMOSA_CLANG_PATH;
#endif
  std::string clang_path =
      pocl_get_string_option("FORMOSA_CLANG", default_clang);
  if (clang_path.empty() && !llvm_path.empty()) {
    clang_path = llvm_path + "/bin/clang";
  }

  char kernel_util_path[POCL_MAX_PATHNAME_LENGTH];
  char errno_stub_path[POCL_MAX_PATHNAME_LENGTH];
  char start_file_path[POCL_MAX_PATHNAME_LENGTH];
  char linker_script_path[POCL_MAX_PATHNAME_LENGTH];
  pocl_get_srcdir_or_datadir(kernel_util_path, "/lib/kernel", "",
                             "/simtsim/kernel_util.cl");
  pocl_get_srcdir_or_datadir(errno_stub_path, "/lib/kernel", "",
                             "/simtsim/errno_stub.c");
  pocl_get_srcdir_or_datadir(start_file_path, "/lib/kernel", "",
                             "/simtsim/simtsim_start.S");
  pocl_get_srcdir_or_datadir(linker_script_path, "/lib/kernel", "",
                             "/simtsim/link.ld");
  std::stringstream ss_out;

  // Link kernel program with predefined kernel functions and libprintf.a
  std::stringstream ss_cmd;
  std::vector<std::string> args = {clang_path,
                                   build_cflags + " -fPIE ",
                                   start_file_path,
                                   bitcode_path,
                                   kernel_util_path,
                                   errno_stub_path,
                                   build_ldflags,
                                   " -T ",
                                   linker_script_path,
                                   " -Wl,-pie -o ",
                                   elf_path};
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
    static int n = 0;
    char dump_path[POCL_MAX_PATHNAME_LENGTH];
    sprintf(dump_path, "program%d.dump", n++);
    std::vector<std::string> args = {objdump_path, "-d", elf_path, ">",
                                     dump_path};
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

int pocl_simtsim_get_elf_name(cl_program program, cl_uint device_i,
                              char *elf_name) {
  if (program == nullptr || elf_name == nullptr)
    return -1;

  char program_bc[POCL_MAX_PATHNAME_LENGTH];
  pocl_cache_program_bc_path(program_bc, program, device_i);

  /* remove extension name */
  char *last_dot = strrchr(program_bc, '.');
  if (last_dot != nullptr)
    *last_dot = '\0';

  strcpy(elf_name, program_bc);
  strncat(elf_name, ".fsa.bin", POCL_MAX_PATHNAME_LENGTH - 1);
  return 0;
}

int pocl_simtsim_check_kernel_valid(uint32_t group_size) {
  /* Skip all check because the simulator can accomodate everything */
  return 0;
}
