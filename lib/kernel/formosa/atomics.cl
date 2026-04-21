/* Formosa-specific old-style atomics backed by RISC-V AMOs.
 */

#define FORMOSA_DEF_AMO_HELPER_AS(TYPE, NAME, ASM, AS)                         \
  static inline TYPE _CL_OVERLOADABLE                                           \
  formosa_##NAME(volatile AS TYPE *p, TYPE val) {                               \
    TYPE old;                                                                   \
    __asm__ volatile(ASM " %0, %2, (%1)"                                        \
                     : "=r"(old)                                                \
                     : "r"(p), "r"(val)                                         \
                     : "memory");                                               \
    return old;                                                                 \
  }

#define FORMOSA_DEF_AMO_HELPER(TYPE, NAME, ASM)                                 \
  FORMOSA_DEF_AMO_HELPER_AS(TYPE, NAME, ASM, __global)                          \
  FORMOSA_DEF_AMO_HELPER_AS(TYPE, NAME, ASM, __local)

#define FORMOSA_DEF_BIN_AND_ALIAS_AS(NAME, TYPE, AS, BODY)                      \
  __attribute__((overloadable)) TYPE atomic_##NAME(volatile AS TYPE *p,         \
                                                   TYPE val) {                  \
    BODY                                                                        \
  }                                                                             \
  __attribute__((overloadable)) TYPE atom_##NAME(volatile AS TYPE *p,           \
                                                 TYPE val) {                    \
    return atomic_##NAME(p, val);                                               \
  }

#define FORMOSA_DEF_BIN_AND_ALIAS(NAME, TYPE, BODY)                             \
  FORMOSA_DEF_BIN_AND_ALIAS_AS(NAME, TYPE, __global, BODY)                      \
  FORMOSA_DEF_BIN_AND_ALIAS_AS(NAME, TYPE, __local, BODY)

#define FORMOSA_DEF_UNARY_AND_ALIAS_AS(NAME, TYPE, AS, BODY)                    \
  __attribute__((overloadable)) TYPE atomic_##NAME(volatile AS TYPE *p) {       \
    BODY                                                                        \
  }                                                                             \
  __attribute__((overloadable)) TYPE atom_##NAME(volatile AS TYPE *p) {         \
    return atomic_##NAME(p);                                                    \
  }

#define FORMOSA_DEF_UNARY_AND_ALIAS(NAME, TYPE, BODY)                           \
  FORMOSA_DEF_UNARY_AND_ALIAS_AS(NAME, TYPE, __global, BODY)                    \
  FORMOSA_DEF_UNARY_AND_ALIAS_AS(NAME, TYPE, __local, BODY)

#define FORMOSA_DEF_NAMED_BIN_AS(FN, TYPE, AS, BODY)                            \
  __attribute__((overloadable)) TYPE FN(volatile AS TYPE *p, TYPE val) {        \
    BODY                                                                        \
  }

#define FORMOSA_DEF_NAMED_BIN(FN, TYPE, BODY)                                   \
  FORMOSA_DEF_NAMED_BIN_AS(FN, TYPE, __global, BODY)                            \
  FORMOSA_DEF_NAMED_BIN_AS(FN, TYPE, __local, BODY)

#define FORMOSA_DEF_INTEGER_ATOMICS(TYPE, ADD_HELPER, XOR_HELPER, AND_HELPER,   \
                                    OR_HELPER, SWAP_HELPER, MIN_HELPER,         \
                                    MAX_HELPER)                                 \
  FORMOSA_DEF_BIN_AND_ALIAS(add, TYPE, return ADD_HELPER(p, val);)              \
  FORMOSA_DEF_BIN_AND_ALIAS(sub, TYPE, return ADD_HELPER(p, (TYPE)(0 - val));)  \
  FORMOSA_DEF_UNARY_AND_ALIAS(inc, TYPE, return ADD_HELPER(p, (TYPE)1);)        \
  FORMOSA_DEF_UNARY_AND_ALIAS(dec, TYPE, return ADD_HELPER(p, (TYPE)-1);)       \
  FORMOSA_DEF_BIN_AND_ALIAS(xor, TYPE, return XOR_HELPER(p, val);)              \
  FORMOSA_DEF_BIN_AND_ALIAS(and, TYPE, return AND_HELPER(p, val);)              \
  FORMOSA_DEF_BIN_AND_ALIAS(or, TYPE, return OR_HELPER(p, val);)                \
  FORMOSA_DEF_BIN_AND_ALIAS(xchg, TYPE, return SWAP_HELPER(p, val);)            \
  FORMOSA_DEF_BIN_AND_ALIAS(min, TYPE, return MIN_HELPER(p, val);)              \
  FORMOSA_DEF_BIN_AND_ALIAS(max, TYPE, return MAX_HELPER(p, val);)

FORMOSA_DEF_AMO_HELPER(int, amoswap_i32, "amoswap.w.aqrl")
FORMOSA_DEF_AMO_HELPER(int, amoadd_i32, "amoadd.w.aqrl")
FORMOSA_DEF_AMO_HELPER(int, amoxor_i32, "amoxor.w.aqrl")
FORMOSA_DEF_AMO_HELPER(int, amoand_i32, "amoand.w.aqrl")
FORMOSA_DEF_AMO_HELPER(int, amoor_i32, "amoor.w.aqrl")
FORMOSA_DEF_AMO_HELPER(int, amomin_i32, "amomin.w.aqrl")
FORMOSA_DEF_AMO_HELPER(int, amomax_i32, "amomax.w.aqrl")
FORMOSA_DEF_AMO_HELPER(uint, amoswap_u32, "amoswap.w.aqrl")
FORMOSA_DEF_AMO_HELPER(uint, amoadd_u32, "amoadd.w.aqrl")
FORMOSA_DEF_AMO_HELPER(uint, amoxor_u32, "amoxor.w.aqrl")
FORMOSA_DEF_AMO_HELPER(uint, amoand_u32, "amoand.w.aqrl")
FORMOSA_DEF_AMO_HELPER(uint, amoor_u32, "amoor.w.aqrl")
FORMOSA_DEF_AMO_HELPER(uint, amominu_u32, "amominu.w.aqrl")
FORMOSA_DEF_AMO_HELPER(uint, amomaxu_u32, "amomaxu.w.aqrl")

#ifdef cl_khr_int64_base_atomics
FORMOSA_DEF_AMO_HELPER(long, amoswap_i64, "amoswap.d.aqrl")
FORMOSA_DEF_AMO_HELPER(long, amoadd_i64, "amoadd.d.aqrl")
FORMOSA_DEF_AMO_HELPER(long, amoxor_i64, "amoxor.d.aqrl")
FORMOSA_DEF_AMO_HELPER(long, amoand_i64, "amoand.d.aqrl")
FORMOSA_DEF_AMO_HELPER(long, amoor_i64, "amoor.d.aqrl")
FORMOSA_DEF_AMO_HELPER(long, amomin_i64, "amomin.d.aqrl")
FORMOSA_DEF_AMO_HELPER(long, amomax_i64, "amomax.d.aqrl")
FORMOSA_DEF_AMO_HELPER(ulong, amoswap_u64, "amoswap.d.aqrl")
FORMOSA_DEF_AMO_HELPER(ulong, amoadd_u64, "amoadd.d.aqrl")
FORMOSA_DEF_AMO_HELPER(ulong, amoxor_u64, "amoxor.d.aqrl")
FORMOSA_DEF_AMO_HELPER(ulong, amoand_u64, "amoand.d.aqrl")
FORMOSA_DEF_AMO_HELPER(ulong, amoor_u64, "amoor.d.aqrl")
FORMOSA_DEF_AMO_HELPER(ulong, amominu_u64, "amominu.d.aqrl")
FORMOSA_DEF_AMO_HELPER(ulong, amomaxu_u64, "amomaxu.d.aqrl")
#endif

FORMOSA_DEF_INTEGER_ATOMICS(int, formosa_amoadd_i32, formosa_amoxor_i32,
                            formosa_amoand_i32, formosa_amoor_i32,
                            formosa_amoswap_i32, formosa_amomin_i32,
                            formosa_amomax_i32)
FORMOSA_DEF_INTEGER_ATOMICS(uint, formosa_amoadd_u32, formosa_amoxor_u32,
                            formosa_amoand_u32, formosa_amoor_u32,
                            formosa_amoswap_u32, formosa_amominu_u32,
                            formosa_amomaxu_u32)

#ifdef cl_khr_int64_base_atomics
FORMOSA_DEF_INTEGER_ATOMICS(long, formosa_amoadd_i64, formosa_amoxor_i64,
                            formosa_amoand_i64, formosa_amoor_i64,
                            formosa_amoswap_i64, formosa_amomin_i64,
                            formosa_amomax_i64)
FORMOSA_DEF_INTEGER_ATOMICS(ulong, formosa_amoadd_u64, formosa_amoxor_u64,
                            formosa_amoand_u64, formosa_amoor_u64,
                            formosa_amoswap_u64, formosa_amominu_u64,
                            formosa_amomaxu_u64)
#endif

FORMOSA_DEF_NAMED_BIN(_cl_atomic_min, int, return formosa_amomin_i32(p, val);)
FORMOSA_DEF_NAMED_BIN(_cl_atomic_min, uint,
                      return formosa_amominu_u32(p, val);)
FORMOSA_DEF_NAMED_BIN(_cl_atomic_max, int, return formosa_amomax_i32(p, val);)
FORMOSA_DEF_NAMED_BIN(_cl_atomic_max, uint,
                      return formosa_amomaxu_u32(p, val);)

#ifdef cl_khr_int64_base_atomics
FORMOSA_DEF_NAMED_BIN(_cl_atom_min, long, return formosa_amomin_i64(p, val);)
FORMOSA_DEF_NAMED_BIN(_cl_atom_min, ulong,
                      return formosa_amominu_u64(p, val);)
FORMOSA_DEF_NAMED_BIN(_cl_atom_max, long, return formosa_amomax_i64(p, val);)
FORMOSA_DEF_NAMED_BIN(_cl_atom_max, ulong,
                      return formosa_amomaxu_u64(p, val);)
#endif

__attribute__((overloadable))
float atomic_xchg(volatile __global float *p, float val)
{
  return as_float(formosa_amoswap_u32((volatile __global uint *)p, as_uint(val)));
}

__attribute__((overloadable))
float atomic_xchg(volatile __local float *p, float val)
{
  return as_float(formosa_amoswap_u32((volatile __local uint *)p, as_uint(val)));
}

__attribute__((overloadable))
float atom_xchg(volatile __global float *p, float val)
{
  return atomic_xchg(p, val);
}

__attribute__((overloadable))
float atom_xchg(volatile __local float *p, float val)
{
  return atomic_xchg(p, val);
}

#ifdef cl_khr_int64_base_atomics
__attribute__((overloadable))
double atomic_xchg(volatile __global double *p, double val)
{
  return as_double(formosa_amoswap_u64((volatile __global ulong *)p,
                                       as_ulong(val)));
}

__attribute__((overloadable))
double atomic_xchg(volatile __local double *p, double val)
{
  return as_double(formosa_amoswap_u64((volatile __local ulong *)p,
                                       as_ulong(val)));
}

__attribute__((overloadable))
double atom_xchg(volatile __global double *p, double val)
{
  return atomic_xchg(p, val);
}

__attribute__((overloadable))
double atom_xchg(volatile __local double *p, double val)
{
  return atomic_xchg(p, val);
}
#endif

#undef FORMOSA_DEF_INTEGER_ATOMICS
#undef FORMOSA_DEF_UNARY_AND_ALIAS
#undef FORMOSA_DEF_UNARY_AND_ALIAS_AS
#undef FORMOSA_DEF_BIN_AND_ALIAS
#undef FORMOSA_DEF_BIN_AND_ALIAS_AS
#undef FORMOSA_DEF_NAMED_BIN
#undef FORMOSA_DEF_NAMED_BIN_AS
#undef FORMOSA_DEF_AMO_HELPER
#undef FORMOSA_DEF_AMO_HELPER_AS
