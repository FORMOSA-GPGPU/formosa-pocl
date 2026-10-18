/* SPDX-FileCopyrightText: 2026 CASLab, National Cheng Kung University
 * SPDX-License-Identifier: MIT
 *
 * Formosa C11 atomic implementation.
 *
 * Global AMOs bypass the SM's non-coherent data cache. Atomic loads and stores
 * must use that same linearization path; ordinary RISC-V loads/stores would
 * observe a different cached copy. Use stronger RMW operations while retaining
 * PoCL's scalar, address-space, memory-order and memory-scope implementations.
 */

#define FORMOSA_ATOMIC_LOAD(T, AT, UAT, AS, AS_TYPE)                      \
  static T _CL_OVERLOADABLE formosa_atomic_load(                          \
      volatile AS AT *p, memory_order order, memory_scope scope) {        \
    return AS_TYPE(                                                       \
        __opencl_atomic_fetch_or((volatile AS UAT *)p, 0, order, scope)); \
  }

#define FORMOSA_ATOMIC_LOAD_AS(AS)                                     \
  FORMOSA_ATOMIC_LOAD(int, atomic_int, atomic_uint, AS, as_int)        \
  FORMOSA_ATOMIC_LOAD(uint, atomic_uint, atomic_uint, AS, as_uint)     \
  FORMOSA_ATOMIC_LOAD(long, atomic_long, atomic_ulong, AS, as_long)    \
  FORMOSA_ATOMIC_LOAD(ulong, atomic_ulong, atomic_ulong, AS, as_ulong) \
  FORMOSA_ATOMIC_LOAD(float, atomic_float, atomic_uint, AS, as_float)  \
  FORMOSA_ATOMIC_LOAD(double, atomic_double, atomic_ulong, AS, as_double)

FORMOSA_ATOMIC_LOAD_AS(__global)
FORMOSA_ATOMIC_LOAD_AS(__local)
FORMOSA_ATOMIC_LOAD_AS(__generic)

#undef FORMOSA_ATOMIC_LOAD_AS
#undef FORMOSA_ATOMIC_LOAD

#define __opencl_atomic_load formosa_atomic_load
#define __opencl_atomic_store(P, V, O, S) \
  ((void)__opencl_atomic_exchange(P, V, O, S))
#include "../svm_atomics_host.cl"
