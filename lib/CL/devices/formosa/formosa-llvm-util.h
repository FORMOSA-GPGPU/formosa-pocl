#ifndef FORMSA_LLVM_UTIL_H
#define FORMSA_LLVM_UTIL_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>

/**
 * Build the LLVM module and write it to the specified bitcode path. The number
 * of kernels and the corresponding names will be returned in the provided
 * pointers.
 * @param LLVMModule The LLVM module to build.
 * @param BitcodePath The path where the bitcode will be written.
 * @param NumKernels Pointer to store the number of kernels found.
 * @param Names Pointer to store the names of the kernels found. The names will
 * be null-terminated and separated by null characters.
 */
void pocl_fsa_build_kernel(void *LLVMModule, char *BitcodePath,
                           unsigned *NumKernels, char **Names);

/**
 * Get the address of a symbol in the ELF file.
 * @param ELFPath The path to the ELF file.
 * @param SymbolName The name of the symbol to find.
 * @return The symbol's address on success. Note that 0 can be a valid
 *         address (e.g. _start is at .org 0x0). On failure returns
 *         UINT64_MAX.
 */
uint64_t pocl_fsa_get_symbol_pc(const char *ELFPath, const char *SymbolName);

#ifdef __cplusplus
}
#endif

#endif  // FORMOSA_LLVM_UTIL_H
