/* OpenCL built-in library: cbrt() for Formosa

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

/*
 * ====================================================
 * Copyright (C) 1993 by Sun Microsystems, Inc. All rights reserved.
 *
 * Developed at SunPro, a Sun Microsystems, Inc. business.
 * Permission to use, copy, modify, and distribute this
 * software is freely granted, provided that this notice
 * is preserved.
 * ====================================================
 *
 * sf_cbrt.c -- float version of s_cbrt.c.
 * Conversion to float by Ian Lance Taylor, Cygnus Support, ian@cygnus.com.
 *
 * Adapted to OpenCL bitcasts and vector overloads from newlib:
 * https://github.com/mirror/newlib-cygwin/blob/16740220a22d09a1c63714d93f1efc5fbe3927f3/newlib/libm/common/sf_cbrt.c
 * https://github.com/mirror/newlib-cygwin/blob/3312f960a70cb1648139502b1d7b85351c6ee621/newlib/libm/common/s_cbrt.c
 */

#include "../templates.h"

#pragma OPENCL FP_CONTRACT OFF

float _CL_OVERLOADABLE
cbrt (float x)
{
  uint bits = as_uint (x);
  uint sign = bits & 0x80000000U;
  uint hx = bits & 0x7fffffffU;
  if (hx >= 0x7f800000U)
    return x + x;
  if (hx == 0)
    return x;

  x = as_float (hx);
  float t;
  if (hx < 0x00800000U)
    {
      t = x * 0x1.0p24f;
      t = as_float (as_uint (t) / 3U + 642849266U);
    }
  else
    t = as_float (hx / 3U + 709958130U);

  float r = t * t / x;
  float s = 5.4285717010e-01f + r * t;
  t *= 3.5714286566e-01f
       + 1.6071428061e+00f / (s + 1.4142856598e+00f - 7.0530611277e-01f / s);
  return as_float (as_uint (t) | sign);
}

IMPLEMENT_BUILTIN_V_V (cbrt, float2, NAME1_2 (cbrt))
IMPLEMENT_BUILTIN_V_V (cbrt, float3, NAME1_3 (cbrt))
IMPLEMENT_BUILTIN_V_V (cbrt, float4, NAME1_4 (cbrt))
IMPLEMENT_BUILTIN_V_V (cbrt, float8, NAME1_8 (cbrt))
IMPLEMENT_BUILTIN_V_V (cbrt, float16, NAME1_16 (cbrt))

#ifdef cl_khr_fp64
double _CL_OVERLOADABLE
cbrt (double x)
{
  ulong bits = as_ulong (x);
  ulong sign = bits & 0x8000000000000000UL;
  bits &= 0x7fffffffffffffffUL;
  uint hx = (uint)(bits >> 32);
  if (hx >= 0x7ff00000U)
    return x + x;
  if (bits == 0)
    return x;

  x = as_double (bits);
  double t;
  if (hx < 0x00100000U)
    {
      t = x * 0x1.0p54;
      ulong scaled = as_ulong (t);
      uint hi = (uint)(scaled >> 32);
      t = as_double ((scaled & 0xffffffffUL)
                     | ((ulong)(hi / 3U + 696219795U) << 32));
    }
  else
    t = as_double ((ulong)(hx / 3U + 715094163U) << 32);

  double r = t * t / x;
  double s = 5.42857142857142815906e-01 + r * t;
  t *= 3.57142857142857150787e-01
       + 1.60714285714285720630e+00
             / (s + 1.41428571428571436819e+00
                - 7.05306122448979611050e-01 / s);
  t = as_double (((as_ulong (t) >> 32) + 1UL) << 32);
  s = t * t;
  r = x / s;
  double w = t + t;
  r = (r - t) / (w + r);
  t = t + t * r;
  return as_double (as_ulong (t) | sign);
}

IMPLEMENT_BUILTIN_V_V (cbrt, double2, NAME1_2 (cbrt))
IMPLEMENT_BUILTIN_V_V (cbrt, double3, NAME1_3 (cbrt))
IMPLEMENT_BUILTIN_V_V (cbrt, double4, NAME1_4 (cbrt))
IMPLEMENT_BUILTIN_V_V (cbrt, double8, NAME1_8 (cbrt))
IMPLEMENT_BUILTIN_V_V (cbrt, double16, NAME1_16 (cbrt))
#endif
