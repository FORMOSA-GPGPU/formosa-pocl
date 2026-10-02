/* OpenCL built-in library: libclc math for Formosa

   Copyright (c) 2026 pocl developers

   Permission is hereby granted, free of charge, to any person obtaining a copy
   of this software and associated documentation files (the "Software"), to
   deal in the Software without restriction, including without limitation the
   rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
   sell copies of the Software, and to permit persons to whom the Software is
   furnished to do so, subject to the following conditions:

   The above copyright notice and this permission notice shall be included in
   all copies or substantial portions of the Software.

   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
   IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
   AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
   FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
   IN THE SOFTWARE.
*/

#include "../templates.h"
#include "../libclc/misc.h"
#include "../libclc/vtables.h"

DEFINE_EXPR_V_VVV (pocl_fma, fma (a, b, c))

/* Keep vector elements sequential to fit the per-thread stack. */
#define VECTOR_OVERLOAD(NAME, TYPE, WIDTH)                                     \
  TYPE##WIDTH _CL_OVERLOADABLE NAME (TYPE##WIDTH x)                              \
  {                                                                          \
    TYPE##WIDTH result;                                                       \
    _Pragma ("clang loop unroll(disable)")                                     \
    for (int i = 0; i < WIDTH; ++i)                                            \
      result[i] = NAME (x[i]);                                                 \
    return result;                                                            \
  }

#define VECTOR_OVERLOADS(NAME, TYPE)                                           \
  VECTOR_OVERLOAD (NAME, TYPE, 2)                                              \
  VECTOR_OVERLOAD (NAME, TYPE, 3)                                              \
  VECTOR_OVERLOAD (NAME, TYPE, 4)                                              \
  VECTOR_OVERLOAD (NAME, TYPE, 8)                                              \
  VECTOR_OVERLOAD (NAME, TYPE, 16)

#define vtype float
#define v2type v2float
#define itype int
#define utype uint
#define as_vtype as_float
#define as_itype as_int
#define as_utype as_uint
#define convert_vtype convert_float
#include "../libclc/acosh_fp32.cl"
#include "../libclc/asinh_fp32.cl"
#include "../libclc/atanh_fp32.cl"
#include "../libclc/log1p_fp32.cl"
#undef vtype
#undef v2type
#undef itype
#undef utype
#undef as_vtype
#undef as_itype
#undef as_utype
#undef convert_vtype

VECTOR_OVERLOADS (log1p, float)
VECTOR_OVERLOADS (acosh, float)
VECTOR_OVERLOADS (asinh, float)
VECTOR_OVERLOADS (atanh, float)

#ifdef cl_khr_fp64
#define vtype double
#define v2type v2double
#define itype long
#define utype ulong
#define as_vtype as_double
#define as_itype as_long
#define as_utype as_ulong
#define convert_vtype convert_double
#define convert_utype convert_ulong
#define convert_uinttype convert_uint
#include "../libclc/ep_log_fp64.cl"
#include "../libclc/log1p_fp64.cl"
#include "../libclc/acosh_fp64.cl"
#include "../libclc/asinh_fp64.cl"
#include "../libclc/atanh_fp64.cl"
#undef vtype
#undef v2type
#undef itype
#undef utype
#undef as_vtype
#undef as_itype
#undef as_utype
#undef convert_vtype
#undef convert_utype
#undef convert_uinttype

VECTOR_OVERLOADS (log1p, double)
VECTOR_OVERLOADS (acosh, double)
VECTOR_OVERLOADS (asinh, double)
VECTOR_OVERLOADS (atanh, double)
#endif

#undef VECTOR_OVERLOADS
#undef VECTOR_OVERLOAD
