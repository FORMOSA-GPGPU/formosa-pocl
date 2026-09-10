/* OpenCL built-in library: lgamma_r()

   Copyright (c) 2026 pocl developers

   Permission is hereby granted, free of charge, to any person obtaining a copy
   of this software and associated documentation files (the "Software"), to deal
   in the Software without restriction, including without limitation the rights
   to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
   copies of the Software, and to permit persons to whom the Software is
   furnished to do so, subject to the following conditions:

   The above copyright notice and this permission notice shall be included in
   all copies or substantial portions of the Software.

   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
   IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
   AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
   OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
   THE SOFTWARE.
*/

#include "templates.h"

/* signp is 0 at zero and negative integers; otherwise sign(Gamma). */

#define IMPLEMENT_LGAMMA_R_SCALAR(STYPE, ZERO, TWO)                           \
  STYPE _CL_OVERLOADABLE lgamma_r (STYPE a, int __private *c)                 \
  {                                                                           \
    STYPE fl = floor (a);                                                     \
    if (a > ZERO)                                                             \
      *c = 1;                                                                 \
    else if (a == fl)                                                         \
      *c = 0;                                                                 \
    else                                                                      \
      {                                                                       \
        STYPE rem = fl - TWO * floor (fl / TWO);                              \
        *c = rem != ZERO ? -1 : 1;                                            \
      }                                                                       \
    return lgamma (a);                                                        \
  }

IMPLEMENT_LGAMMA_R_SCALAR (float, 0.0f, 2.0f)
IMPLEMENT_BUILTIN_V_VPJ_ADDRSPACE (lgamma_r, float, int, __local)
IMPLEMENT_BUILTIN_V_VPJ_ADDRSPACE (lgamma_r, float, int, __global)
IF_GEN_AS (IMPLEMENT_BUILTIN_V_VPJ_ADDRSPACE (lgamma_r, float, int, __generic))
IMPLEMENT_BUILTIN_V_VPJ (lgamma_r, float2, int2, int, int, lo, hi)
IMPLEMENT_BUILTIN_V_VPJ (lgamma_r, float3, int3, int2, int, lo, s2)
IMPLEMENT_BUILTIN_V_VPJ (lgamma_r, float4, int4, int2, int2, lo, hi)
IMPLEMENT_BUILTIN_V_VPJ (lgamma_r, float8, int8, int4, int4, lo, hi)
IMPLEMENT_BUILTIN_V_VPJ (lgamma_r, float16, int16, int8, int8, lo, hi)

__IF_FP64 (
IMPLEMENT_LGAMMA_R_SCALAR (double, 0.0, 2.0)
IMPLEMENT_BUILTIN_V_VPJ_ADDRSPACE (lgamma_r, double, int, __local)
IMPLEMENT_BUILTIN_V_VPJ_ADDRSPACE (lgamma_r, double, int, __global)
IF_GEN_AS (IMPLEMENT_BUILTIN_V_VPJ_ADDRSPACE (lgamma_r, double, int, __generic))
IMPLEMENT_BUILTIN_V_VPJ (lgamma_r, double2, int2, int, int, lo, hi)
IMPLEMENT_BUILTIN_V_VPJ (lgamma_r, double3, int3, int2, int, lo, s2)
IMPLEMENT_BUILTIN_V_VPJ (lgamma_r, double4, int4, int2, int2, lo, hi)
IMPLEMENT_BUILTIN_V_VPJ (lgamma_r, double8, int8, int4, int4, lo, hi)
IMPLEMENT_BUILTIN_V_VPJ (lgamma_r, double16, int16, int8, int8, lo, hi))
