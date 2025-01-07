#include "pocl-formosa-util.h"

#include <libcomm/comm.h>
#include <libcomm/msg.h>
#include <semaphore.h>
#include <signal.h>

#include <cstdio>
#include <fstream>
#include <iostream>
#include <sstream>

#include "casvp-config/casvp-config.h"
#include "falloc/fsa_mem_allocator.h"
#include "pocl-formosa-util.h"
#include "pocl.h"
#include "pocl_cl.h"
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
#include <llvm/Object/ELFObjectFile.h>
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
  msg_t *msg = msg_create(buffer_data->msg_id++, WRITE, size,
                          buffer_data->buf_address + dst_offset);
  if (msg == nullptr) return -1;
  msg = msg_set_payload(msg, static_cast<const uint8_t *>(host_ptr));
  if (msg == nullptr) return -1;
  int status = ipc_send_write_msg(buffer_data->client_fd, msg);
  msg_destroy(msg);
  return status;
}

int fsa_copy_from_dev(formosa_buffer_data_t *buffer_data, void *host_ptr,
                      uint64_t src_offset, size_t size) {
  if (buffer_data == nullptr || host_ptr == nullptr) return -1;
  if ((src_offset + size) > buffer_data->buf_size) return -1;
  msg_t *msg = msg_create(buffer_data->msg_id++, READ, size,
                          buffer_data->buf_address + src_offset);
  if (msg == nullptr) return -1;
  int status = ipc_send_read_msg(buffer_data->client_fd, msg);
  if (status != 0) goto FSA_COPY_FROM_DEV_FINALLY;
  memcpy(host_ptr, msg->payload, size);
FSA_COPY_FROM_DEV_FINALLY:
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
  sem_init(&sem, 0, 0);
  sem_wait(&sem);    // Wait until the signal handler post the sem.
  uint64_t ack = 0;  // acknowledge from device
  int status = -1;
  status = fsa_read_csr(dd, CASVP_FORMOSA_CSR_ACK, &ack);
  if (ack != 1) return -1;
  if (status != 0) return -1;
  sem_destroy(&sem);
  return 0;
}

int fsa_write_csr(pocl_formosa_data_t *dd, uint64_t addr, uint64_t value) {
  if (dd == nullptr) return -1;
  msg_t *msg = msg_create(dd->msg_id++, WRITE, 8, addr);
  if (msg == nullptr) return -1;
  msg = msg_set_payload(msg, reinterpret_cast<uint8_t *>(&value));
  if (msg == nullptr) return -1;
  int status = ipc_send_write_msg(dd->client_fd, msg);
  msg_destroy(msg);
  return status;
}

int fsa_read_csr(pocl_formosa_data_t *dd, uint64_t addr, uint64_t *value) {
  if (dd == nullptr) return -1;
  msg_t *msg = msg_create(dd->msg_id++, READ, 8, addr);
  if (msg == nullptr) return -1;
  int status = ipc_send_read_msg(dd->client_fd, msg);
  if (status != 0) goto FSA_READ_CSR_FINALLY;
  *value = *reinterpret_cast<uint64_t *>(msg->payload);
FSA_READ_CSR_FINALLY:
  msg_destroy(msg);
  return status;
}

void fsa_int_handler(int sig) {
  if (sig == SIGUSR1) {
    if (sem_post(&sem) < 0) {
      std::cerr << "Received unexpected device interrupt.";
    };
  }
}

static int exec(const char *cmd, std::ostream &out) {
  char buffer[128];
  auto pipe = popen(cmd, "r");
  if (!pipe) {
    // throw std::runtime_error("popen() failed!");
    return -1;
  }
  while (!feof(pipe)) {
    if (fgets(buffer, 128, pipe) != nullptr) out << buffer;
  }
  return pclose(pipe);
}

static void createTrampolineFunction(
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

static void generateTrampolineForKernels(
    llvm::SmallVector<std::string, 8> &funcNames, llvm::Module *module) {
  llvm::SmallVector<llvm::Function *, 8> functionsToErase;
  for (auto &function : module->functions()) {
    if (!pocl::isKernelToProcess(function)) continue;
    createTrampolineFunction(&function, module, funcNames);
  }
}

static char *convertToCharArray(
    const llvm::SmallVector<std::string, 8> &names) {
  // Calculate the total length required for the buffer
  size_t totalLength = 0;
  for (const auto &name : names) {
    totalLength += name.size() + 1;  // +1 for the null terminator
  }

  // Allocate buffer
  char *buffer = (char *)malloc(totalLength * sizeof(char));
  if (buffer == nullptr) {
    std::cerr << "Memory allocation failed" << std::endl;
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

int fsa_compile_program(char **kernel_names, int *num_kernels,
                        char *str_program_fsa_bin, void *llvm_module) {
  int err;
  std::string llvm_path = FORMOSA_LLVM;
  std::string llvm_objdump_path = llvm_path + "/bin/llvm-objdump";
  std::string build_cflags = "-mcpu=formosa-gpgpu ";
  std::string extra_cflags = pocl_get_string_option("POCL_FORMOSA_CFLAGS", "");
  if (extra_cflags == "") {
    POCL_MSG_WARN(
        "Environment variable 'POCL_FORMOSA_CFLAGS' is not set, default to "
        "-O1 -mllvm -fsa-pdom-level\n");
    build_cflags += "-O1 -mllvm -fsa-pdom-level";
  } else {
    build_cflags += extra_cflags;
  }

  std::string build_ldflags = "";
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
  generateTrampolineForKernels(kernelNames, module);

  *num_kernels = kernelNames.size();
  *kernel_names = convertToCharArray(kernelNames);

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
  pocl_get_srcdir_or_datadir(kernel_util_path, "/lib/CL/devices", "",
                             "/formosa/kernel_util.cl");
  pocl_get_srcdir_or_datadir(start_file_path, "/lib/CL/devices", "",
                             "/formosa/start.S");

  std::stringstream ss_cmd, ss_out;
  std::vector<std::string> args = {
      clang_path,      build_cflags,  bitcode_path, kernel_util_path,
      start_file_path, build_ldflags, "-o",         elf_path};
  ss_cmd = generate_command(args);
  POCL_MSG_PRINT_LLVM("running \"%s\"\n", ss_cmd.str().c_str());
  err = exec(ss_cmd.str().c_str(), ss_out);
  if (err != 0) {
    POCL_MSG_ERR("%s\n", ss_out.str().c_str());
    return err;
  }

  if (POCL_DEBUGGING_ON) {
    std::string objdump_path(llvm_objdump_path);

    std::stringstream ss_cmd, ss_out;
    std::vector<std::string> args = {objdump_path, "-D", elf_path, ">",
                                     "program.dump"};
    ss_cmd = generate_command(args);
    POCL_MSG_PRINT_LLVM("running \"%s\"\n", ss_cmd.str().c_str());
    err = exec(ss_cmd.str().c_str(), ss_out);
    if (err != 0) {
      POCL_MSG_ERR("%s\n", ss_out.str().c_str());
      return err;
    }
  }
  return 0;
}

uint64_t fsa_get_trampoline_pc(const char *elf_path, const char *kernel_name) {
  auto bufferOrError = llvm::MemoryBuffer::getFile(std::string(elf_path));
  if (!bufferOrError) {
    POCL_ABORT("Error: failed to open ELF file %s\n", elf_path);
  }

  // Parse ELF file
  auto objOrError = llvm::object::ObjectFile::createELFObjectFile(
      bufferOrError.get()->getMemBufferRef());
  if (!objOrError) {
    POCL_ABORT("Error: failed to parse ELF file %s\n", elf_path);
  }

  std::unique_ptr<llvm::object::ObjectFile> obj = std::move(objOrError.get());
  for (const llvm::object::SymbolRef &symbol : obj->symbols()) {
    llvm::Expected<llvm::object::SymbolRef::Type> typeOrError =
        symbol.getType();
    if (!typeOrError) {
      POCL_ABORT("Error: failed to get symbol type\n");
    }
    if (*typeOrError != llvm::object::SymbolRef::ST_Function) continue;

    llvm::Expected<llvm::StringRef> nameOrError = symbol.getName();
    if (!nameOrError) {
      POCL_ABORT("Error: failed to get symbol name\n");
    }
    if (nameOrError.get().str() == kernel_name) {
      llvm::Expected<uint64_t> addrOrError = symbol.getAddress();
      if (!addrOrError) {
        POCL_ABORT("Error: failed to get symbol address\n");
      }
      POCL_MSG_PRINT_LLVM("Found symbol %s at 0x%lx\n", kernel_name,
                          addrOrError.get());
      return addrOrError.get();
    }
  }
  POCL_ABORT("Error: failed to find symbol %s\n", kernel_name);
  return 0;
}
