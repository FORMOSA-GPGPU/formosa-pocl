#include "pocl-formosa-util.h"

#include <libcomm/comm.h>
#include <libcomm/msg.h>

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

#include <llvm/Analysis/TargetLibraryInfo.h>
#include <llvm/Analysis/TargetTransformInfo.h>
#include <llvm/Bitcode/BitcodeWriter.h>
#include <llvm/IR/Module.h>
#include <llvm/Support/FileSystem.h>
#include <llvm/Support/raw_ostream.h>

#include <string>
#include <system_error>

#include "LLVMUtils.h"
#include "llvm/Bitcode/BitcodeReader.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Metadata.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Linker/Linker.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/Alignment.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetOptions.h"
#include "llvm/Transforms/IPO.h"
#include "llvm/Transforms/Utils/Cloning.h"

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

// Store function arguments in a single argument buffer.
static bool createArgumentsBuffer(
    llvm::Function *function, llvm::Module *module,
    llvm::SmallVector<std::string, 8> &funcNames) {
  auto &Context = module->getContext();
  const llvm::DataLayout &DL = module->getDataLayout();

  std::string TargetTriple = module->getTargetTriple();

  auto I32Ty = llvm::Type::getInt32Ty(Context);
  auto I8Ty = llvm::Type::getInt8Ty(Context);
  auto I8PtrTy = I8Ty->getPointerTo();

  // Create new function signature: func(i8*) {return old
  // func->getReturnType()}, 3rd arg(false) here means argument list is not
  // variable(i.e. the passed arg cnt is not changable)
  auto NewFuncType =
      llvm::FunctionType::get(function->getReturnType(), {I8PtrTy}, false);
  auto NewFunc = llvm::Function::Create(NewFuncType, function->getLinkage(),
                                        function->getName() + "_formosa");
  module->getFunctionList().insert(function->getIterator(), NewFunc);
  NewFunc->takeName(function);

  auto EntryBlock = llvm::BasicBlock::Create(Context, "entry", NewFunc);
  llvm::IRBuilder<> Builder(EntryBlock);

  // Access function arguments
  auto ai = NewFunc->arg_begin();
  auto ArgBuffer = &*ai++;
  ArgBuffer->setName("ArgBuffer");

  unsigned arg_idx = 0;
  unsigned arg_offset = 0;

  llvm::Value *allocated_local_mem = nullptr;

  for (auto &OldArg : function->args()) {
    auto ArgType = OldArg.getType();
    auto ArgOffset = llvm::ConstantInt::get(I32Ty, arg_offset);
    llvm::Value *Arg;
    if (pocl::isLocalMemFunctionArg(function, arg_idx)) {
      if (allocated_local_mem == nullptr) {
        // Load __local_size
        auto local_size_ptr =
            Builder.CreateGEP(I8Ty, ArgBuffer, ArgOffset, "__local_size_ptr");
        auto local_size =
            Builder.CreateLoad(I32Ty, local_size_ptr, "__local_size");
        arg_offset += 8;
        // Call vx_local_alloc(__local_size)
        auto function_type = llvm::FunctionType::get(I8PtrTy, {I32Ty}, false);
        auto fsa_local_alloc_func =
            module->getOrInsertFunction("fsa_local_alloc", function_type);
        allocated_local_mem = Builder.CreateCall(fsa_local_alloc_func,
                                                 {local_size}, "__local_mem");
      }
      // Load argument __offset
      auto offset_ptr = Builder.CreateGEP(I8Ty, ArgBuffer, ArgOffset,
                                          OldArg.getName() + "_offset_ptr");
      auto offset =
          Builder.CreateLoad(I32Ty, offset_ptr, OldArg.getName() + "_offset");
      arg_offset += 8;
      // Apply pointer offset
      Arg = Builder.CreateGEP(I8PtrTy, allocated_local_mem, offset,
                              OldArg.getName() + "_byte_ptr");
    } else {
      auto offset_ptr = Builder.CreateGEP(I8Ty, ArgBuffer, ArgOffset,
                                          OldArg.getName() + "_offset_ptr");
      Arg =
          Builder.CreateLoad(ArgType, offset_ptr, OldArg.getName() + "_loaded");
      arg_offset += 8;
    }
    auto instr = llvm::cast<llvm::Instruction>(Arg);
    assert(instr != nullptr);
    OldArg.replaceAllUsesWith(Arg);
    arg_idx += 1;
  }

  // Move the body of the old function to the new function
  NewFunc->splice(NewFunc->end(), function);

  // Connect the entry block to the first block of the old function
  for (auto &BB : *NewFunc) {
    if (&BB != EntryBlock) {
      Builder.CreateBr(&BB);
      break;
    }
  }

  funcNames.push_back(NewFunc->getName().str());

  return true;
}

static void generateArgumentBufferForKernels(
    llvm::SmallVector<std::string, 8> &funcNames, llvm::Module *module) {
  llvm::SmallVector<llvm::Function *, 8> functionsToErase;
  for (auto &function : module->functions()) {
    if (!pocl::isKernelToProcess(function)) continue;
    if (createArgumentsBuffer(&function, module, funcNames))
      functionsToErase.push_back(&function);
  }
  for (auto function : functionsToErase) {
    function->eraseFromParent();
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

int fsa_compile_program(char **kernel_names, int *num_kernels,
                        char *str_program_fsa_bin, void *llvm_module) {
  int err;
  std::string llvm_path = FORMOSA_LLVM;
  std::string llvm_objdump_path = llvm_path + "/bin/llvm-objdump";
  std::string build_cflags = pocl_get_string_option("POCL_FORMOSA_CFLAGS", "");
  if (build_cflags == "") {
    POCL_MSG_WARN(
        "Environment variable 'POCL_FORMOSA_CFLAGS' is not set, default to "
        "-mcpu=formosa-gpgpu -O1 -mllvm -fsa-pdom-level\n");
    build_cflags = "-mcpu=formosa-gpgpu -O1 -mllvm -fsa-pdom-level";
  }

  std::string build_ldflags =
      pocl_get_string_option("POCL_FORMOSA_LDFLAGS", "");
  if (build_ldflags == "") {
    POCL_MSG_WARN("Environment variable 'POCL_FORMOSA_LDFLAGS' is not set\n");
    build_ldflags = "-fuse-ld=lld -nostartfiles";
  }

  char bitcode_path[POCL_MAX_PATHNAME_LENGTH];
  err = pocl_mk_tempname(bitcode_path, "/tmp/pocl_formosa_program", ".bc",
                         nullptr);
  if (err != 0) return err;

  char elf_path[POCL_MAX_PATHNAME_LENGTH];
  memcpy(elf_path, str_program_fsa_bin, strlen(str_program_fsa_bin) + 1);

  auto module = (llvm::Module *)llvm_module;
  llvm::SmallVector<std::string, 8> kernelNames;
  generateArgumentBufferForKernels(kernelNames, module);

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

  char kernel_main_path[POCL_MAX_PATHNAME_LENGTH];
  pocl_get_srcdir_or_datadir(kernel_main_path, "/lib/CL/devices", "",
                             "/formosa/kernel_main.cl");

  std::stringstream ss_cmd, ss_out;
  ss_cmd << clang_path.c_str() << " " << build_cflags << " " << bitcode_path
         << " " << kernel_main_path << " " << build_ldflags << " -o "
         << elf_path;
  POCL_MSG_PRINT_LLVM("running \"%s\"\n", ss_cmd.str().c_str());
  err = exec(ss_cmd.str().c_str(), ss_out);
  if (err != 0) {
    POCL_MSG_ERR("%s\n", ss_out.str().c_str());
    return err;
  }

  if (POCL_DEBUGGING_ON) {
    std::string objdump_path(llvm_objdump_path);

    std::stringstream ss_cmd, ss_out;
    ss_cmd << objdump_path.c_str() << " -D " << elf_path << " > program.dump";

    POCL_MSG_PRINT_LLVM("running \"%s\"\n", ss_cmd.str().c_str());
    err = exec(ss_cmd.str().c_str(), ss_out);
    if (err != 0) {
      POCL_MSG_ERR("%s\n", ss_out.str().c_str());
      return err;
    }
  }

  POCL_MSG_PRINT_LLVM("running \"%s\"\n", ss_cmd.str().c_str());
  err = exec(ss_cmd.str().c_str(), ss_out);
  if (err != 0) {
    POCL_MSG_ERR("%s\n", ss_out.str().c_str());
    return err;
  }

  return 0;
}
