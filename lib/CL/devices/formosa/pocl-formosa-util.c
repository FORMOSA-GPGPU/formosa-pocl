#include "pocl_debug.h"
#include "pocl-formosa-util.h"
int compile_formosa_program(char**kernel_names, int* num_kernels, char* sz_program_vxbin, void* llvm_module) {
  int err;

  const char* llvm_install_path = getenv("LLVM_PREFIX");
  if (llvm_install_path) {
    if (!pocl_exists(llvm_install_path)) {
      POCL_MSG_ERR("$LLVM_PREFIX: '%s' doesn't exist\n", llvm_install_path);
      return -1;
    }
    POCL_MSG_PRINT_INFO("using $LLVM_PREFIX=%s!\n", llvm_install_path);
  }

  std::string build_cflags = pocl_get_string_option("POCL_FORMOSA_CFLAGS", "");
  if (build_cflags == "") {
    POCL_MSG_WARN("Environment variable 'POCL_FORMOSA_CFLAGS' is not set\n");
    // return -1;
  }

  std::string build_ldflags = pocl_get_string_option("POCL_FORMOSA_LDFLAGS", "");
  if(build_ldflags == ""){
    POCL_MSG_WARN("Environment variable 'POCL_FORMOSA_LDFLAGS' is not set\n");
    // return -1;
  }

  char sz_program_bc[POCL_MAX_PATHNAME_LENGTH + 1];
  err = pocl_mk_tempname(sz_program_bc, "~/tmp/pocl_formosa_program", ".bc", nullptr);
  if (err != 0)
    return err;

  char sz_program_elf[POCL_MAX_PATHNAME_LENGTH + 1];
  err = pocl_mk_tempname(sz_program_elf, "~/tmp/pocl_formosa_program", ".elf", nullptr);
  if (err != 0)
    return err;

  auto module = (llvm::Module *)llvm_module;
  llvm::SmallVector<std::string, 8> kernelNames;
  processKernels(kernelNames, module);
  addKernelSelect(kernelNames, module);

  *num_kernels = kernelNames.size();
  *kernel_names = convertToCharArray(kernelNames);

  {
    std::error_code EC;
    llvm::raw_fd_ostream file(sz_program_bc, EC, llvm::sys::fs::OF_None);
    llvm::WriteBitcodeToFile(*module, file);
    file.close();
  }

  if (POCL_DEBUGGING_ON) {
    std::error_code EC;
    llvm::raw_fd_ostream file("program.ll", EC, llvm::sys::fs::OF_None);
    module->print(file, nullptr);
    file.close();
  }

  {
    std::string clang_path(CLANG);
    if (llvm_install_path) {
      clang_path.replace(0, strlen(LLVM_PREFIX), llvm_install_path);
    }

    char sz_kernel_main[POCL_MAX_PATHNAME_LENGTH];
    pocl_get_srcdir_or_datadir (sz_kernel_main, "/lib/CL/devices", "", "/formosa/kernel_main.c");

    std::stringstream ss_cmd, ss_out;
    ss_cmd << clang_path.c_str() << " " << build_cflags << " " << sz_program_bc << " " << sz_kernel_main << " " << build_ldflags << " -o " << sz_program_elf;
    POCL_MSG_PRINT_LLVM("running \"%s\"\n", ss_cmd.str().c_str());
    int err = exec(ss_cmd.str().c_str(), ss_out);
    if (err != 0) {
      POCL_MSG_ERR("%s\n", ss_out.str().c_str());
      return err;
    }
  }

  if (POCL_DEBUGGING_ON) {
    std::string objdump_path(LLVM_OBJDUMP);
    if (llvm_install_path) {
      objdump_path.replace(0, strlen(LLVM_PREFIX), llvm_install_path);
    }

    std::stringstream ss_cmd, ss_out;
    ss_cmd << objdump_path.c_str() << " -D " << sz_program_elf << " > program.dump";

    POCL_MSG_PRINT_LLVM("running \"%s\"\n", ss_cmd.str().c_str());
    int err = exec(ss_cmd.str().c_str(), ss_out);
    if (err != 0) {
      POCL_MSG_ERR("%s\n", ss_out.str().c_str());
      return err;
    }
  }

  {
    std::string vxbintool_path = pocl_get_string_option ("POCL_FORMOSA_BINTOOL", "");
    if (vxbintool_path == ""){
      POCL_MSG_WARN("Environment 'POCL_FORMOSA_BINTOOL' is not set\n");
      // return -1;
    }
    std::stringstream ss_cmd, ss_out;
    ss_cmd << vxbintool_path << " " << sz_program_elf << " " << sz_program_vxbin;
    POCL_MSG_PRINT_LLVM("running \"%s\"\n", ss_cmd.str().c_str());
    int err = exec(ss_cmd.str().c_str(), ss_out);
    if (err != 0) {
      POCL_MSG_ERR("%s\n", ss_out.str().c_str());
      return err;
    }
  }

  return 0;
}