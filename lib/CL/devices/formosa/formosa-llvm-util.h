#ifndef FORMSA_LLVM_UTIL_H
#define FORMSA_LLVM_UTIL_H

#include "pocl-formosa-internal.h"

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>

/**
 * Build the LLVM module and write it to the specified bitcode path. Kernel
 * names and argument layouts are returned in the backend's program data.
 * @param LLVMModule The LLVM module to build.
 * @param BitcodePath The path where the bitcode will be written.
 * @param ProgramData Receives kernel names and their matching argument layouts.
 * @return CL_SUCCESS on success, otherwise an OpenCL error code.
 */
int pocl_fsa_build_kernel(void *LLVMModule, char *BitcodePath,
                          formosa_program_data_t *ProgramData);

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
