/* OpenCL built-in library: lgamma() for Formosa

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

#include "../templates.h"

float _CL_OVERLOADABLE
lgamma (float a)
{
  int s;
  return lgamma_r (a, &s);
}

IMPLEMENT_BUILTIN_V_V (lgamma, float2, NAME1_2 (lgamma))
IMPLEMENT_BUILTIN_V_V (lgamma, float3, NAME1_3 (lgamma))
IMPLEMENT_BUILTIN_V_V (lgamma, float4, NAME1_4 (lgamma))
IMPLEMENT_BUILTIN_V_V (lgamma, float8, NAME1_8 (lgamma))
IMPLEMENT_BUILTIN_V_V (lgamma, float16, NAME1_16 (lgamma))

__IF_FP64 (
double _CL_OVERLOADABLE
lgamma (double a)
{
  int s;
  return lgamma_r (a, &s);
}

IMPLEMENT_BUILTIN_V_V (lgamma, double2, NAME1_2 (lgamma))
IMPLEMENT_BUILTIN_V_V (lgamma, double3, NAME1_3 (lgamma))
IMPLEMENT_BUILTIN_V_V (lgamma, double4, NAME1_4 (lgamma))
IMPLEMENT_BUILTIN_V_V (lgamma, double8, NAME1_8 (lgamma))
IMPLEMENT_BUILTIN_V_V (lgamma, double16, NAME1_16 (lgamma)))
