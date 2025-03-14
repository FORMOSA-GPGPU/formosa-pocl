#include "pocl-formosa-util.h"
#include "formosa-driver.h"

#include <libcomm/comm.h>
#include <libcomm/msg.h>
#include <semaphore.h>
#include <signal.h>
#include <linux/elf.h>

#include <cstdint>
#include <iostream>
#include <sstream>
#include <string>
#include <tuple>

#include "casvp-config.h"
#include "formosa-driver.h"
#include "pocl-formosa-util.h"
#include "pocl.h"
#include "pocl_cache.h"
#include "pocl_debug.h"
#include "pocl_file_util.h"
#include "pocl_runtime_config.h"
#include "pocl_util.h"

#if LLVM_MAJOR >= 17
#include <llvm/Transforms/IPO/Internalize.h>
#endif

#include <LLVMUtils.h>
#include <llvm/Analysis/TargetLibraryInfo.h>
#include <llvm/Analysis/TargetTransformInfo.h>
#include <llvm/Bitcode/BitcodeReader.h>
#include <llvm/Bitcode/BitcodeWriter.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/Instructions.h>
#include <llvm/IR/LegacyPassManager.h>
#include <llvm/IR/Metadata.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/Type.h>
#include <llvm/IR/Verifier.h>
#include <llvm/Linker/Linker.h>
// #include <llvm/Object/ELF.h>
// #include <llvm/Object/ELFObjectFile.h>
#include <llvm/Object/ObjectFile.h>
#include <llvm/Object/SymbolicFile.h>
#include <llvm/Support/FileSystem.h>
#include <llvm/Support/MemoryBuffer.h>
#include <llvm/Support/raw_ostream.h>
#include <llvm/Target/TargetMachine.h>
#include <llvm/Transforms/Utils/Cloning.h>

#include <string>
#include <system_error>

static sem_t sem;
int fsa_check_occupancy(uint32_t group_size, uint32_t *max_local_mem) {
  // check group size
  uint32_t warps_per_core = CASVP_FORMOSA_WARPS_PER_CORE;
  uint32_t threads_per_warp = CASVP_FORMOSA_THREADS_PER_WARP;
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
    uint64_t local_mem_size = CASVP_FORMOSA_LOCAL_MEM_SIZE;
    *max_local_mem = local_mem_size / groups_per_core;
  }

  return 0;
}

int fsa_get_elf_name(cl_program program, cl_uint device_i, char *elf_name) {
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

int fsa_upload_kernel(const char *elf_file, pocl_formosa_data_t *dd, uint64_t *kernel_dev_addr, uint64_t *kernel_base) {
  if (elf_file == nullptr || dd == nullptr) return -1;
  uint64_t kernel_size = 0;
  uint64_t min_base = -1ULL;
  FILE *elf = fopen(elf_file, "rb");
  rewind(elf);
  Elf64_Ehdr ehdr;
  fread(&ehdr, sizeof(ehdr), 1, elf);
  // 1st pass, iterate all program headers to find the minimum base address
  for (int i = 0; i < ehdr.e_phnum; i++) {
    fseek(elf, ehdr.e_phoff + i * (ehdr.e_phentsize), SEEK_SET);
    Elf64_Phdr phdr;
    // read prog header from elf
    fread(&phdr, sizeof(phdr), 1, elf);
    if (phdr.p_type != PT_LOAD || phdr.p_paddr < FSA_GLOBAL_MEM_BASE)
        continue;
    uint64_t addr = phdr.p_paddr;
    if(addr < min_base)
      min_base = addr;
    if(addr + phdr.p_memsz > kernel_size)
      kernel_size = addr + phdr.p_memsz;
  }
  // calculate absolute kernel size (minus min_base as offset)
  kernel_size -= min_base;
  void *kernel_start_addr_;
  fsa_malloc(&kernel_start_addr_, kernel_size);
  uint64_t kernel_start_addr = (uint64_t)kernel_start_addr_;
  *kernel_base = min_base;
  *kernel_dev_addr = (uint64_t)kernel_start_addr;
  if (POCL_DEBUGGING_ON) {
    printf("kernel_start_addr: %lx\n", (uint64_t)kernel_start_addr);
    printf("kernel_size: %lx\n", kernel_size);
  }

  // move file pointer to beginning of file
  rewind(elf);
  uint8_t *host_ptr = (uint8_t*)calloc(1, sizeof(uint8_t) * kernel_size);
  uint64_t offset = 0;

  // 2nd pass, copy all program headers to device memory
  for (int i = 0; i < ehdr.e_phnum; i++) {
    fseek(elf, ehdr.e_phoff + i * (ehdr.e_phentsize), SEEK_SET);
    Elf64_Phdr phdr;
    fread(&phdr, sizeof(phdr), 1, elf);
    if (phdr.p_type != PT_LOAD || phdr.p_paddr < FSA_GLOBAL_MEM_BASE)
        continue;
    uint64_t size = phdr.p_filesz;
    uint64_t offset = phdr.p_paddr - min_base;
    fseek(elf, phdr.p_offset, SEEK_SET);
    if(size) {
      fread(host_ptr + offset, size, 1, elf);
    } else {
      size = phdr.p_memsz;
    }
    printf("Copy %lx to %lx with size %lx\n",
      (uint64_t)host_ptr + offset, (uint64_t)kernel_start_addr + offset, size);
    fsa_copy_to_dev((uint64_t)kernel_start_addr + offset, host_ptr + offset, size);
  }
  free(host_ptr);
  fclose(elf);
  return 0;
}

int fsa_wait_ack(pocl_formosa_data_t *dd) {
  if (dd == nullptr) return -1;
  sem_init(&sem, 0, 0);
  sem_wait(&sem);    // Wait until the signal handler post the sem.
  uint64_t ack = 0;  // acknowledge from device
  int err = fsa_mmio(CASVP_FORMOSA_CSR_ACK, 0, &ack);
  if (err != 0) {
    POCL_MSG_ERR("Failed to read acknowledge from device (%d)\n", err);
    return -1;
  }
  if (ack != 0) {
    POCL_MSG_ERR("Unexpected acknowledge from device (%ld)\n", ack);
    return -1;
  }

  uint64_t status = 0;
  uint64_t ecid, ewid, mcause, mepc, mtval;
  err = fsa_mmio(CASVP_FORMOSA_CSR_STATUS, 0, &status);
  if (err != 0) {
    POCL_MSG_ERR("Failed to read status from device (%d)\n", err);
    return -1;
  }
  switch (status) {
    default:
    case 0x0:  // Okay
      break;
    case 0x1:  // Bad dimension
      POCL_MSG_ERR("Bad dimension\n");
      break;
    case 0x2:  // Exceptions
      err = fsa_mmio(CASVP_FORMOSA_CSR_ECID, 0, &ecid);
      err |= fsa_mmio(CASVP_FORMOSA_CSR_EWID, 0, &ewid);
      err |= fsa_mmio(CASVP_FORMOSA_CSR_MCAUSE, 0, &mcause);
      err |= fsa_mmio(CASVP_FORMOSA_CSR_MEPC, 0, &mepc);
      err |= fsa_mmio(CASVP_FORMOSA_CSR_MTVAL, 0, &mtval);
      if (err != 0) {
        POCL_MSG_ERR("Failed to read exception information from device\n");
        break;
      }
      POCL_MSG_ERR(
          "\nException occurs:\n"
          "\tecid:   0x%08lx\n"
          "\tewid:   0x%08lx\n"
          "\tmcause: 0x%08lx\n"
          "\tmepc:   0x%08lx\n"
          "\tmtval:  0x%08lx\n",
          ecid, ewid, mcause, mepc, mtval);
      break;
  }

  err = fsa_mmio(CASVP_FORMOSA_CSR_ACK, 1, nullptr);
  if (err != 0) {
    POCL_MSG_ERR("Failed to read acknowledge from device (%d)\n", err);
    return -1;
  }
  sem_destroy(&sem);
  return (status == 0) ? 0 : -1;
}

void fsa_int_handler(int sig) {
  if (sig == SIGUSR1) {
    if (sem_post(&sem) < 0) {
      POCL_MSG_ERR("Received unexpected device interrupt.\n");
    }
  }
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

static void create_trampoline_function(
    llvm::Function *function, llvm::Module *module,
    llvm::SmallVector<std::string, 8> &funcNames) {
  auto &context = module->getContext();
  llvm::IRBuilder<> builder(context);

  // Get the function's argument types
  llvm::FunctionType *funcType = function->getFunctionType();
  llvm::ArrayRef<llvm::Type *> argTypes = funcType->params();

  // Create a structure type for the arguments
  llvm::StructType *argStructType =
      llvm::StructType::create(context, argTypes, "ArgStruct");

  // Create the trampoline function type: `void trampoline(i8*)`
  llvm::Type *voidType = llvm::Type::getVoidTy(context);
  auto i8Type = llvm::Type::getInt8Ty(context);
  auto i32Type = llvm::Type::getInt32Ty(context);
  auto i8PtrType = i8Type->getPointerTo();
  llvm::FunctionType *trampolineType =
      llvm::FunctionType::get(voidType, {i8PtrType}, false);

  // Create the trampoline function
  auto trampolineFunction =
      llvm::Function::Create(trampolineType, llvm::Function::ExternalLinkage,
                             function->getName() + "_trampoline", module);

  // Create a basic block in the trampoline function
  auto entryBlock =
      llvm::BasicBlock::Create(context, "entry", trampolineFunction);
  builder.SetInsertPoint(entryBlock);

  // Get the trampoline's argument (the `i8*` pointer)
  auto argPtr = trampolineFunction->getArg(0);

  // Cast the `i8*` pointer to the structure type representing the arguments
  llvm::Type *argStructPtrType = argStructType->getPointerTo();
  auto castedArg =
      builder.CreateBitCast(argPtr, argStructPtrType, "casted_arg");

  // Extract each argument from the structure
  std::vector<llvm::Value *> extractedArgs;
  llvm::Value *allocated_local_mem = nullptr;
  for (unsigned i = 0; i < argTypes.size(); ++i) {
    if (pocl::isLocalMemFunctionArg(function, i)) {
      if (allocated_local_mem == nullptr) {
        // Load __local_size
        auto local_size_ptr =
            builder.CreateGEP(i8Type, castedArg, builder.getInt32(0),
                              "arg" + std::to_string(i) + "__local_size_gep");
        auto local_size =
            builder.CreateLoad(i32Type, local_size_ptr,
                               "arg" + std::to_string(i) + "__local_size");
        auto function_type =
            llvm::FunctionType::get(i8PtrType, {i32Type}, false);
        auto fsa_local_alloc_func =
            module->getOrInsertFunction("fsa_local_alloc", function_type);
        allocated_local_mem =
            builder.CreateCall(fsa_local_alloc_func, {local_size},
                               "arg" + std::to_string(i) + "__local_mem");
      }
      // Load argument __offset
      auto offset_name = "arg" + std::to_string(i) + "__offset";
      auto offset_ptr = builder.CreateGEP(
          i8Type, castedArg, builder.getInt32(8), offset_name + "_gep");
      auto offset = builder.CreateLoad(i8Type, offset_ptr, offset_name);
      // Apply pointer offset
      auto byte_ptr = builder.CreateGEP(i8PtrType, allocated_local_mem, offset,
                                        "arg" + std::to_string(i) + "__gep");
      extractedArgs.push_back(byte_ptr);
    } else {
      auto argGEP = builder.CreateStructGEP(
          argStructType, castedArg, i, "arg" + std::to_string(i) + "__gep");
      auto argValue = builder.CreateLoad(argTypes[i], argGEP,
                                         "arg" + std::to_string(i) + "__value");
      extractedArgs.push_back(argValue);
    }
  }

  // Call the target function with the extracted arguments
  auto callInst = builder.CreateCall(function, extractedArgs);

  // Handle return type (if void, return void)
  if (funcType->getReturnType()->isVoidTy()) {
    builder.CreateRetVoid();
  } else {
    llvm::Type *retType = funcType->getReturnType();
    llvm::Value *retValue = callInst;
    // If necessary, process retValue before return
    builder.CreateRet(retValue);
  }

  trampolineFunction->addFnAttr(llvm::Attribute::NoInline);
  trampolineFunction->addFnAttr(llvm::Attribute::OptimizeNone);

  funcNames.push_back(function->getName().str());

  // Finish
  llvm::verifyFunction(*trampolineFunction);
}

static void generate_trampoline_for_kernels(
    llvm::SmallVector<std::string, 8> &funcNames, llvm::Module *module) {
  llvm::SmallVector<llvm::Function *, 8> functionsToErase;
  for (auto &function : module->functions()) {
    if (!pocl::isKernelToProcess(function)) continue;
    create_trampoline_function(&function, module, funcNames);
  }
}

static char *convert_to_char_array(
    const llvm::SmallVector<std::string, 8> &names) {
  // Calculate the total length required for the buffer
  size_t totalLength = 0;
  for (const auto &name : names) {
    totalLength += name.size() + 1;  // +1 for the null terminator
  }

  // Allocate buffer
  char *buffer = new char[totalLength];
  if (buffer == nullptr) {
    POCL_MSG_ERR("Host memory allocation failed\n");
    return nullptr;
  }

  // Copy names into buffer with null separation
  size_t position = 0;
  for (const auto &name : names) {
    std::strcpy(buffer + position, name.c_str());
    position += name.size();
    buffer[position] = '\0';  // Null terminator
    position += 1;
  }

  return buffer;
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

int fsa_compile_program(char **kernel_names, int *num_kernels,
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

  auto module = (llvm::Module *)llvm_module;
  llvm::SmallVector<std::string, 8> kernelNames;
  generate_trampoline_for_kernels(kernelNames, module);

  *num_kernels = kernelNames.size();
  *kernel_names = convert_to_char_array(kernelNames);

  std::error_code EC;
  llvm::raw_fd_ostream file(bitcode_path, EC, llvm::sys::fs::OF_None);
  llvm::WriteBitcodeToFile(*module, file);
  file.close();

  if (POCL_DEBUGGING_ON) {
    std::error_code EC;
    llvm::raw_fd_ostream file("program.ll", EC, llvm::sys::fs::OF_None);
    module->print(file, nullptr);
    file.close();
  }

  std::string clang_path(CLANG);
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
                                         clang_path, build_cflags);
  if (err != 0) {
    POCL_MSG_ERR("%s\n", ss_out.str().c_str());
    return err;
  }
  std::tie(err, ss_out) = compile_source(putchar_src_path, putchar_obj_path,
                                         clang_path, build_cflags);
  if (err != 0) {
    POCL_MSG_ERR("%s\n", ss_out.str().c_str());
    return err;
  }

  // Archive printf.o and putchar.o into libprintf.a
  std::stringstream ss_cmd;
  std::string archive_path = llvm_path + "/bin/llvm-ar";
  std::vector<std::string> args = {archive_path, "rcs", printf_lib_path,
                                   printf_obj_path, putchar_obj_path};
  ss_cmd = generate_command(args);
  POCL_MSG_PRINT_LLVM("Running \"%s\"\n", ss_cmd.str().c_str());
  err = exec(ss_cmd.str().c_str(), ss_out);
  if (err != 0) {
    POCL_MSG_ERR("%s\n", ss_out.str().c_str());
    return err;
  }

  // Link kernel program with predefined kernel functions and libprintf.a
  args = {clang_path,         build_cflags,  start_file_path,
          bitcode_path,       "-L/tmp",      "-lprintf",
          kernel_util_path,   build_ldflags, "-T",
          linker_script_path, "-o",          elf_path};
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

uint64_t fsa_get_symbol_pc(const char *elf_path, const char *symbol_name) {
  auto bufferOrError = llvm::MemoryBuffer::getFile(std::string(elf_path));
  if (!bufferOrError) {
    POCL_ABORT("ERROR (fsa_get_symbol_pc): Failed to open ELF file %s\n",
               elf_path);
  }

  // Parse ELF file
  auto objOrError = llvm::object::ObjectFile::createELFObjectFile(
      bufferOrError.get()->getMemBufferRef());
  if (!objOrError) {
    POCL_ABORT("ERROR (fsa_get_symbol_pc): Failed to parse ELF file %s\n",
               elf_path);
  }

  std::unique_ptr<llvm::object::ObjectFile> obj = std::move(objOrError.get());
  for (const llvm::object::SymbolRef &symbol : obj->symbols()) {
    llvm::Expected<llvm::object::SymbolRef::Type> typeOrError =
        symbol.getType();
    if (!typeOrError) {
      POCL_ABORT("ERROR (fsa_get_symbol_pc): Failed to get symbol type\n");
    }

    llvm::Expected<llvm::StringRef> nameOrError = symbol.getName();
    if (!nameOrError) {
      POCL_ABORT("ERROR (fsa_get_symbol_pc): Failed to get symbol name\n");
    }
    if (nameOrError.get().str() == symbol_name) {
      llvm::Expected<uint64_t> addrOrError = symbol.getAddress();
      if (!addrOrError) {
        POCL_ABORT("ERROR (fsa_get_symbol_pc): Failed to get symbol address\n");
      }
      POCL_MSG_PRINT_LLVM("Found symbol %s at 0x%lx\n", symbol_name,
                          addrOrError.get());
      return addrOrError.get();
    }
  }
  POCL_ABORT("ERROR (fsa_get_symbol_pc): Failed to find symbol %s\n",
             symbol_name);
  return 0;
}


uint64_t fsa_get_symbol_offset(const char *elf_path, const char *symbol_name) {
  if(elf_path == nullptr || symbol_name == nullptr) {
    POCL_ABORT("ERROR (fsa_get_symbol_offset): Null ptr in argument\n");
  }

  auto bufferOrError = llvm::MemoryBuffer::getFile(std::string(elf_path));
  if (!bufferOrError) {
    POCL_ABORT("ERROR (fsa_get_symbol_offset): Failed to open ELF file %s\n",
               elf_path);
  }

  // Parse ELF file
  auto objOrError = llvm::object::ObjectFile::createELFObjectFile(
      bufferOrError.get()->getMemBufferRef());
  if (!objOrError) {
    POCL_ABORT("ERROR (fsa_get_symbol_offset): Failed to parse ELF file %s\n",
               elf_path);
  }
  
  bool found_load = false;
  std::unique_ptr<llvm::object::ObjectFile> obj = std::move(objOrError.get());
  
  uint64_t symbol_addr = fsa_get_symbol_pc(elf_path, symbol_name);
  uint64_t base_addr = obj->getStartAddress().get();
  uint64_t symbol_offset = symbol_addr - base_addr;
  printf("base addr 0x%lx, symbol addr 0x%lx, symbol offset 0x%lx\n", base_addr, symbol_addr, symbol_offset);
  return symbol_offset;
}
