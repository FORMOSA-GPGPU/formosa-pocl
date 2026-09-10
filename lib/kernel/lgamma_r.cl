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

/* Software lgamma: Clang has no lgamma intrinsic, so __builtin_lgammaf
 * becomes an unresolved lgammaf libcall. Stirling + reflection uses log/sin. */

static float
lgamma_positivef (float x)
{
  float acc = 0.0f;
  while (x < 8.0f)
    {
      acc -= log (x);
      x += 1.0f;
    }
  float r = 1.0f / x;
  float r2 = r * r;
  return acc + (x - 0.5f) * log (x) - x + 0.9189385332046727f
         + r * (0.08333333333333333f
                + r2 * (-0.002777777777777778f
                        + r2 * 0.0007936507936507937f));
}

static double
lgamma_positived (double x)
{
  double acc = 0.0;
  while (x < 8.0)
    {
      acc -= log (x);
      x += 1.0;
    }
  double r = 1.0 / x;
  double r2 = r * r;
  return acc + (x - 0.5) * log (x) - x + 0.91893853320467274178
         + r * (0.083333333333333333
                + r2 * (-0.002777777777777778
                        + r2 * 0.0007936507936507937));
}

float _CL_OVERLOADABLE
lgamma_r (float a, int __private *c)
{
  if (isnan (a))
    {
      *c = 0;
      return a;
    }
  if (isinf (a))
    {
      *c = a > 0.0f ? 1 : 0;
      return INFINITY;
    }

  float fl = floor (a);
  if (a > 0.0f)
    *c = 1;
  else if (a == fl)
    {
      *c = 0;
      return INFINITY;
    }
  else
    {
      float rem = fl - 2.0f * floor (fl / 2.0f);
      *c = rem != 0.0f ? -1 : 1;
    }

  if (a > 0.0f)
    return lgamma_positivef (a);

  float t = sin (a * M_PI_F);
  return log (M_PI_F / fabs (t * a)) - lgamma_positivef (-a);
}

IMPLEMENT_BUILTIN_V_VPJ_ADDRSPACE (lgamma_r, float, int, __local)
IMPLEMENT_BUILTIN_V_VPJ_ADDRSPACE (lgamma_r, float, int, __global)
IF_GEN_AS (IMPLEMENT_BUILTIN_V_VPJ_ADDRSPACE (lgamma_r, float, int, __generic))
IMPLEMENT_BUILTIN_V_VPJ (lgamma_r, float2, int2, int, int, lo, hi)
IMPLEMENT_BUILTIN_V_VPJ (lgamma_r, float3, int3, int2, int, lo, s2)
IMPLEMENT_BUILTIN_V_VPJ (lgamma_r, float4, int4, int2, int2, lo, hi)
IMPLEMENT_BUILTIN_V_VPJ (lgamma_r, float8, int8, int4, int4, lo, hi)
IMPLEMENT_BUILTIN_V_VPJ (lgamma_r, float16, int16, int8, int8, lo, hi)

__IF_FP64 (
double _CL_OVERLOADABLE
lgamma_r (double a, int __private *c)
{
  if (isnan (a))
    {
      *c = 0;
      return a;
    }
  if (isinf (a))
    {
      *c = a > 0.0 ? 1 : 0;
      return INFINITY;
    }

  double fl = floor (a);
  if (a > 0.0)
    *c = 1;
  else if (a == fl)
    {
      *c = 0;
      return INFINITY;
    }
  else
    {
      double rem = fl - 2.0 * floor (fl / 2.0);
      *c = rem != 0.0 ? -1 : 1;
    }

  if (a > 0.0)
    return lgamma_positived (a);

  double t = sin (a * M_PI);
  return log (M_PI / fabs (t * a)) - lgamma_positived (-a);
}

IMPLEMENT_BUILTIN_V_VPJ_ADDRSPACE (lgamma_r, double, int, __local)
IMPLEMENT_BUILTIN_V_VPJ_ADDRSPACE (lgamma_r, double, int, __global)
IF_GEN_AS (IMPLEMENT_BUILTIN_V_VPJ_ADDRSPACE (lgamma_r, double, int, __generic))
IMPLEMENT_BUILTIN_V_VPJ (lgamma_r, double2, int2, int, int, lo, hi)
IMPLEMENT_BUILTIN_V_VPJ (lgamma_r, double3, int3, int2, int, lo, s2)
IMPLEMENT_BUILTIN_V_VPJ (lgamma_r, double4, int4, int2, int2, lo, hi)
IMPLEMENT_BUILTIN_V_VPJ (lgamma_r, double8, int8, int4, int4, lo, hi)
IMPLEMENT_BUILTIN_V_VPJ (lgamma_r, double16, int16, int8, int8, lo, hi))
