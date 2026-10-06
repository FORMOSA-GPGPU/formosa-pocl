#include "formosa-llvm-util.h"

#include <cstdlib>

#include "pocl_debug.h"

#if LLVM_MAJOR >= 17
#include <llvm/Transforms/IPO/Internalize.h>
#endif

#include <LLVMUtils.h>
#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/SmallVector.h>
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
#include <llvm/Object/ObjectFile.h>
#include <llvm/Object/SymbolicFile.h>
#include <llvm/Support/Error.h>
#include <llvm/Support/FileSystem.h>
#include <llvm/Support/MathExtras.h>
#include <llvm/Support/MemoryBuffer.h>
#include <llvm/Support/raw_ostream.h>
#include <llvm/Target/TargetMachine.h>
#include <llvm/Transforms/Utils/Cloning.h>

namespace {
int createArgumentLayout(llvm::Function *F, llvm::Module *M,
                         size_t ArgBufferOffset,
                         formosa_kernel_layout_t &Layout) {
  const auto &DL = M->getDataLayout();
  Layout.num_args = F->arg_size();
  Layout.args = static_cast<formosa_kernel_arg_layout_t *>(
      calloc(Layout.num_args, sizeof(*Layout.args)));
  if (Layout.num_args != 0 && Layout.args == nullptr)
    return CL_OUT_OF_HOST_MEMORY;

  // Firmware has already consumed printf metadata. The trampoline argument
  // starts with the 64-bit total local-memory size, followed by kernel args.
  // Align in the complete allocation: a printf prefix can leave the
  // trampoline pointer only 8-byte aligned, even for a 16-byte byval object.
  size_t Offset = ArgBufferOffset + sizeof(uint64_t);
  Layout.alignment = sizeof(uint64_t);
  for (auto &Arg : F->args()) {
    auto *StorageTy =
        Arg.hasByValAttr() ? Arg.getParamByValType() : Arg.getType();
    if (pocl::isLocalMemFunctionArg(F, Arg.getArgNo()))
      StorageTy = llvm::Type::getInt64Ty(M->getContext());
    auto Alignment = DL.getABITypeAlign(StorageTy);
    if (Arg.hasByValAttr() && Arg.getParamAlign())
      Alignment = *Arg.getParamAlign();
    Layout.alignment = std::max(Layout.alignment, size_t(Alignment.value()));
    auto &Slot = Layout.args[Arg.getArgNo()];
    Offset = llvm::alignTo(Offset, Alignment);
    Slot.offset = Offset - ArgBufferOffset;
    Slot.size = DL.getTypeAllocSize(StorageTy).getFixedValue();
    Offset += Slot.size;
  }
  Layout.size = Offset - ArgBufferOffset;
  return CL_SUCCESS;
}

bool createTrampolineFunction(llvm::Function *F, llvm::Module *M,
                              const formosa_kernel_layout_t &Layout,
                              llvm::SmallVector<std::string, 8> &FuncNames) {
  auto &Context = M->getContext();
  llvm::IRBuilder<> Builder(Context);

  // Get the function's argument types
  llvm::FunctionType *FuncType = F->getFunctionType();
  llvm::ArrayRef<llvm::Type *> ArgTypes = FuncType->params();

  // Create the trampoline function type: `void trampoline(i8*)`
  llvm::Type *VoidTy = llvm::Type::getVoidTy(Context);
  auto I8Ty = llvm::Type::getInt8Ty(Context);
  auto I8PtrTy = llvm::PointerType::get(I8Ty->getContext(), 0);
  auto I64Ty = llvm::Type::getInt64Ty(Context);
  llvm::FunctionType *TrampolineTy =
      llvm::FunctionType::get(VoidTy, {I8PtrTy}, false);

  // Create the trampoline function
  auto TrampolineFunction =
      llvm::Function::Create(TrampolineTy, llvm::Function::ExternalLinkage,
                             F->getName() + "_trampolined", M);

  // Create a basic block in the trampoline function
  auto EntryBlock =
      llvm::BasicBlock::Create(Context, "entry", TrampolineFunction);
  Builder.SetInsertPoint(EntryBlock);

  // Get the trampoline's argument (the `i8*` pointer)
  auto ArgPtr = TrampolineFunction->getArg(0);

  auto FTY = llvm::FunctionType::get(I8PtrTy, {}, false);
  auto FSALocalAllocFunc = M->getOrInsertFunction("fsa_local_alloc", FTY);
  llvm::Value *LocalMemPtr =
      Builder.CreateCall(FSALocalAllocFunc, {}, "allocated_local_mem");

  // Use the same byte offsets that the host uses to pack this kernel.
  std::vector<llvm::Value *> ExtractedArgs;
  for (unsigned i = 0; i < ArgTypes.size(); ++i) {
    auto ArgGEP =
        Builder.CreateGEP(I8Ty, ArgPtr, Builder.getInt64(Layout.args[i].offset),
                          "arg" + std::to_string(i) + "__gep");
    if (pocl::isLocalMemFunctionArg(F, i)) {
      // Load argument __offset
      auto OffsetName = "arg" + std::to_string(i) + "__offset";
      auto OffsetValue = Builder.CreateLoad(I64Ty, ArgGEP, OffsetName);
      // Apply pointer offset
      auto OffsetByteGEP =
          Builder.CreateGEP(I8Ty, LocalMemPtr, OffsetValue,
                            "arg" + std::to_string(i) + "_lmem__gep");
      ExtractedArgs.push_back(OffsetByteGEP);
    } else if (F->getArg(i)->hasByValAttr()) {
      // The buffer contains the object itself, not a pointer to it. Keep
      // byval on the call so inlining creates a private copy per work-item.
      ExtractedArgs.push_back(ArgGEP);
    } else {
      auto ArgValue = Builder.CreateLoad(ArgTypes[i], ArgGEP,
                                         "arg" + std::to_string(i) + "__value");
      ExtractedArgs.push_back(ArgValue);
    }
  }

  // Call the target function with the extracted arguments and finish the
  // wrapper CFG before inlining. InlineFunction rewrites the call site and may
  // split blocks, so appending the return after inlining can leave malformed
  // CFG.
  auto CallInst = Builder.CreateCall(F, ExtractedArgs);
  CallInst->setCallingConv(F->getCallingConv());
  for (unsigned i = 0; i < ArgTypes.size(); ++i)
    for (auto Attr : F->getAttributes().getParamAttrs(i))
      CallInst->addParamAttr(i, Attr);

  if (FuncType->getReturnType()->isVoidTy()) {
    Builder.CreateRetVoid();
  } else {
    Builder.CreateRet(CallInst);
  }

  // Inline the kernel into the trampoline to avoid the kernel calling
  // convention overhead.
  bool HadNoInline = F->hasFnAttribute(llvm::Attribute::NoInline);
  bool HadOptimizeNone = F->hasFnAttribute(llvm::Attribute::OptimizeNone);
  F->removeFnAttr(llvm::Attribute::NoInline);
  F->removeFnAttr(llvm::Attribute::OptimizeNone);
  CallInst->removeFnAttr(llvm::Attribute::NoInline);

  llvm::InlineFunctionInfo IFI;
  llvm::InlineResult InlineRes = llvm::InlineFunction(*CallInst, IFI);
  if (!InlineRes.isSuccess()) {
    if (HadNoInline) F->addFnAttr(llvm::Attribute::NoInline);
    if (HadOptimizeNone) F->addFnAttr(llvm::Attribute::OptimizeNone);
    POCL_MSG_WARN("Failed to inline %s into trampoline: %s\n",
                  F->getName().str().c_str(), InlineRes.getFailureReason());
  }

  FuncNames.push_back(F->getName().str());

  // Finish
  llvm::verifyFunction(*TrampolineFunction);
  return InlineRes.isSuccess();
}

int generateTrampolineForKernels(llvm::SmallVector<std::string, 8> &FuncNames,
                                 llvm::Module *M,
                                 formosa_program_data_t *ProgramData) {
  llvm::SmallVector<llvm::Function *, 8> Kernels;
  for (auto &F : M->functions())
    if (pocl::isKernelToProcess(F)) Kernels.push_back(&F);
  ProgramData->num_kernels = Kernels.size();
  ProgramData->kernel_layouts = static_cast<formosa_kernel_layout_t *>(
      calloc(Kernels.size(), sizeof(*ProgramData->kernel_layouts)));
  if (!Kernels.empty() && ProgramData->kernel_layouts == nullptr)
    return CL_OUT_OF_HOST_MEMORY;

  llvm::SmallVector<llvm::Function *, 8> FunctionsToErase;
  for (unsigned i = 0; i < Kernels.size(); ++i) {
    auto *F = Kernels[i];
    auto &Layout = ProgramData->kernel_layouts[i];
    int Err =
        createArgumentLayout(F, M, ProgramData->arg_buffer_offset, Layout);
    if (Err != CL_SUCCESS) return Err;
    bool Inlined = createTrampolineFunction(F, M, Layout, FuncNames);
    if (!Inlined) continue;
    if (F->use_empty())
      FunctionsToErase.push_back(F);
    else
      F->setLinkage(llvm::GlobalValue::InternalLinkage);
  }
  // remove original functions if inlined to save code size
  for (auto *F : FunctionsToErase) F->eraseFromParent();
  return CL_SUCCESS;
}

char *convertToCharArray(const llvm::SmallVector<std::string, 8> &Names) {
  // Calculate the total length required for the buffer
  size_t TotalLength = 0;
  for (const auto &Name : Names) {
    TotalLength += Name.size() + 1;  // +1 for the null terminator
  }

  // Allocate buffer
  char *Buffer = static_cast<char *>(malloc(TotalLength ? TotalLength : 1));
  if (Buffer == nullptr) {
    POCL_MSG_ERR("Host memory allocation failed\n");
    return nullptr;
  }

  // Copy names into buffer with null separation
  size_t Position = 0;
  for (const auto &Name : Names) {
    std::copy(Name.begin(), Name.end(), Buffer + Position);
    Position += Name.size();
    Buffer[Position] = '\0';  // Null terminator
    Position += 1;
  }

  return Buffer;
}

}  // namespace

int pocl_fsa_build_kernel(void *LLVMModule, char *BitcodePath,
                          formosa_program_data_t *ProgramData) {
  auto M = (llvm::Module *)LLVMModule;
  llvm::SmallVector<std::string, 8> KernelNames;
  int Err = generateTrampolineForKernels(KernelNames, M, ProgramData);
  if (Err != CL_SUCCESS) return Err;
  ProgramData->kernel_names = convertToCharArray(KernelNames);
  if (ProgramData->kernel_names == nullptr) return CL_OUT_OF_HOST_MEMORY;

  std::error_code EC;
  llvm::raw_fd_ostream File(BitcodePath, EC, llvm::sys::fs::OF_None);
  llvm::WriteBitcodeToFile(*M, File);
  File.close();

  if (POCL_DEBUGGING_ON) {
    std::error_code EC;
    llvm::raw_fd_ostream File("program.ll", EC, llvm::sys::fs::OF_None);
    M->print(File, nullptr);
    File.close();
  }
  return CL_SUCCESS;
}

uint64_t pocl_fsa_get_symbol_pc(const char *ELFPath, const char *SymbolName) {
  /* 0 is a valid symbol address (Formosa kernels place _start at .org 0x0).
   * Use UINT64_MAX as the not-found / error sentinel. */
  if (ELFPath == nullptr || SymbolName == nullptr) {
    POCL_MSG_ERR("pocl_fsa_get_symbol_pc: invalid arguments\n");
    return UINT64_MAX;
  }

  auto BufferOrError = llvm::MemoryBuffer::getFile(std::string(ELFPath));
  if (!BufferOrError) {
    POCL_MSG_ERR("pocl_fsa_get_symbol_pc: failed to open ELF file %s\n",
                 ELFPath);
    return UINT64_MAX;
  }

  auto ObjOrError = llvm::object::ObjectFile::createELFObjectFile(
      BufferOrError.get()->getMemBufferRef());
  if (!ObjOrError) {
    POCL_MSG_ERR("pocl_fsa_get_symbol_pc: failed to parse ELF file %s\n",
                 ELFPath);
    llvm::consumeError(ObjOrError.takeError());
    return UINT64_MAX;
  }

  std::unique_ptr<llvm::object::ObjectFile> Obj = std::move(ObjOrError.get());
  for (const llvm::object::SymbolRef &Symbol : Obj->symbols()) {
    llvm::Expected<llvm::object::SymbolRef::Type> TypeOrError =
        Symbol.getType();
    if (!TypeOrError) {
      POCL_MSG_ERR("pocl_fsa_get_symbol_pc: failed to get symbol type\n");
      llvm::consumeError(TypeOrError.takeError());
      return UINT64_MAX;
    }

    llvm::Expected<llvm::StringRef> NameOrError = Symbol.getName();
    if (!NameOrError) {
      POCL_MSG_ERR("pocl_fsa_get_symbol_pc: failed to get symbol name\n");
      llvm::consumeError(NameOrError.takeError());
      return UINT64_MAX;
    }
    if (NameOrError.get().str() == SymbolName) {
      llvm::Expected<uint64_t> AddrOrError = Symbol.getAddress();
      if (!AddrOrError) {
        POCL_MSG_ERR(
            "pocl_fsa_get_symbol_pc: failed to get address for symbol %s\n",
            SymbolName);
        llvm::consumeError(AddrOrError.takeError());
        return UINT64_MAX;
      }
      POCL_MSG_PRINT_LLVM("Found symbol %s at 0x%lx\n", SymbolName,
                          AddrOrError.get());
      return AddrOrError.get();
    }
  }
  POCL_MSG_ERR("pocl_fsa_get_symbol_pc: symbol %s not found in %s\n",
               SymbolName, ELFPath);
  return UINT64_MAX;
}
