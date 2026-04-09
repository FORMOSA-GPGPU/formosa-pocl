#include "formosa-llvm-util.h"

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
#include <llvm/Support/FileSystem.h>
#include <llvm/Support/MemoryBuffer.h>
#include <llvm/Support/raw_ostream.h>
#include <llvm/Target/TargetMachine.h>
#include <llvm/Transforms/Utils/Cloning.h>

namespace {
void createTrampolineFunction(llvm::Function *F, llvm::Module *M,
                              llvm::SmallVector<std::string, 8> &FuncNames) {
  auto &Context = M->getContext();
  llvm::IRBuilder<> Builder(Context);

  // Get the function's argument types
  llvm::FunctionType *FuncType = F->getFunctionType();
  llvm::ArrayRef<llvm::Type *> ArgTypes = FuncType->params();

  // Create a structure type for the arguments
  llvm::StructType *ArgStructType =
      llvm::StructType::create(Context, ArgTypes, "ArgStruct");

  // Create the trampoline function type: `void trampoline(i8*)`
  llvm::Type *VoidTy = llvm::Type::getVoidTy(Context);
  auto I8Ty = llvm::Type::getInt8Ty(Context);
  auto I32Ty = llvm::Type::getInt32Ty(Context);
  auto I8PtrTy = llvm::PointerType::get(I8Ty->getContext(), 0);
  auto I64Ty = llvm::Type::getInt64Ty(Context);
  llvm::FunctionType *TrampolineTy =
      llvm::FunctionType::get(VoidTy, {I8PtrTy}, false);

  // Create the trampoline function
  auto TrampolineFunction =
      llvm::Function::Create(TrampolineTy, llvm::Function::ExternalLinkage,
                             F->getName() + "_trampoline", M);

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

  // Cast the `i8*` pointer to the structure type representing the arguments
  llvm::Type *ArgStructPtrType = llvm::PointerType::get(ArgStructType->getContext(), 0);
  auto ArgStructBytes = Builder.CreateGEP(I8Ty, ArgPtr, Builder.getInt32(8));
  auto CastedArg = Builder.CreateBitCast(ArgStructBytes, ArgStructPtrType);

  // Extract each argument from the structure
  std::vector<llvm::Value *> ExtractedArgs;
  for (unsigned i = 0; i < ArgTypes.size(); ++i) {
    if (pocl::isLocalMemFunctionArg(F, i)) {
      // Load argument __offset
      auto OffsetName = "arg" + std::to_string(i) + "__offset";
      auto OffsetGEP = Builder.CreateStructGEP(
          ArgStructType, CastedArg, i, "arg" + std::to_string(i) + "__gep");
      auto OffsetValue = Builder.CreateLoad(I64Ty, OffsetGEP, OffsetName);
      // Apply pointer offset
      auto OffsetByteGEP =
          Builder.CreateGEP(I8Ty, LocalMemPtr, OffsetValue,
                            "arg" + std::to_string(i) + "_lmem__gep");
      ExtractedArgs.push_back(OffsetByteGEP);
    } else {
      auto ArgGEP = Builder.CreateStructGEP(
          ArgStructType, CastedArg, i, "arg" + std::to_string(i) + "__gep");
      auto ArgValue = Builder.CreateLoad(ArgTypes[i], ArgGEP,
                                         "arg" + std::to_string(i) + "__value");
      ExtractedArgs.push_back(ArgValue);
    }
  }

  // Call the target function with the extracted arguments
  auto CallInst = Builder.CreateCall(F, ExtractedArgs);

  // Handle return type (if void, return void)
  if (FuncType->getReturnType()->isVoidTy()) {
    Builder.CreateRetVoid();
  } else {
    llvm::Type *RetType = FuncType->getReturnType();
    llvm::Value *RetValue = CallInst;
    // If necessary, process RetValue before return
    Builder.CreateRet(RetValue);
  }

  TrampolineFunction->addFnAttr(llvm::Attribute::NoInline);
  TrampolineFunction->addFnAttr(llvm::Attribute::OptimizeNone);

  FuncNames.push_back(F->getName().str());

  // Finish
  llvm::verifyFunction(*TrampolineFunction);
}

void generateTrampolineForKernels(llvm::SmallVector<std::string, 8> &FuncNames,
                                  llvm::Module *M) {
  llvm::SmallVector<llvm::Function *, 8> FunctionsToErase;
  for (auto &F : M->functions()) {
    if (!pocl::isKernelToProcess(F)) continue;
    createTrampolineFunction(&F, M, FuncNames);
  }
}

char *convertToCharArray(const llvm::SmallVector<std::string, 8> &Names) {
  // Calculate the total length required for the buffer
  size_t TotalLength = 0;
  for (const auto &Name : Names) {
    TotalLength += Name.size() + 1;  // +1 for the null terminator
  }

  // Allocate buffer
  char *Buffer = new char[TotalLength];
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

void pocl_fsa_build_kernel(void *LLVMModule, char *BitcodePath,
                           unsigned *NumKernels, char **Names) {
  auto M = (llvm::Module *)LLVMModule;
  llvm::SmallVector<std::string, 8> KernelNames;
  generateTrampolineForKernels(KernelNames, M);

  *NumKernels = KernelNames.size();
  *Names = convertToCharArray(KernelNames);

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
}

uint64_t pocl_fsa_get_symbol_pc(const char *ELFPath, const char *SymbolName) {
  auto BufferOrError = llvm::MemoryBuffer::getFile(std::string(ELFPath));
  if (!BufferOrError) {
    POCL_ABORT("ERROR (pocl_fsa_get_symbol_pc): Failed to open ELF file %s\n",
               ELFPath);
  }

  // Parse ELF file
  auto ObjOrError = llvm::object::ObjectFile::createELFObjectFile(
      BufferOrError.get()->getMemBufferRef());
  if (!ObjOrError) {
    POCL_ABORT("ERROR (pocl_fsa_get_symbol_pc): Failed to parse ELF file %s\n",
               ELFPath);
  }

  std::unique_ptr<llvm::object::ObjectFile> Obj = std::move(ObjOrError.get());
  for (const llvm::object::SymbolRef &Symbol : Obj->symbols()) {
    llvm::Expected<llvm::object::SymbolRef::Type> TypeOrError =
        Symbol.getType();
    if (!TypeOrError) {
      POCL_ABORT("ERROR (pocl_fsa_get_symbol_pc): Failed to get symbol type\n");
    }

    llvm::Expected<llvm::StringRef> NameOrError = Symbol.getName();
    if (!NameOrError) {
      POCL_ABORT("ERROR (pocl_fsa_get_symbol_pc): Failed to get symbol name\n");
    }
    if (NameOrError.get().str() == SymbolName) {
      llvm::Expected<uint64_t> AddrOrError = Symbol.getAddress();
      if (!AddrOrError) {
        POCL_ABORT(
            "ERROR (pocl_fsa_get_symbol_pc): Failed to get symbol address\n");
      }
      POCL_MSG_PRINT_LLVM("Found symbol %s at 0x%lx\n", SymbolName,
                          AddrOrError.get());
      return AddrOrError.get();
    }
  }
  POCL_ABORT("ERROR (pocl_fsa_get_symbol_pc): Failed to find symbol %s\n",
             SymbolName);
  return 0;
}
