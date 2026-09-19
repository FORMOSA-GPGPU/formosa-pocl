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

   This implementation is adapted from ROCm Device Libraries' OCML
   lib/kernel/ocml/src/lgamma_rF.cl.  The PoCL-specific changes to this file are
   distributed under the MIT license.  The adapted OCML portions retain the
   following license and the original Sun notice below.

   University of Illinois/NCSA Open Source License

   Copyright (c) 2014-2016, Advanced Micro Devices, Inc.
   All rights reserved.

   Developed by:

       AMD Research and AMD HSA Software Development
       Advanced Micro Devices, Inc.
       www.amd.com

   Permission is hereby granted, free of charge, to any person obtaining a copy
   of this software and associated documentation files (the "Software"), to
   deal with the Software without restriction, including without limitation
   the rights to use, copy, modify, merge, publish, distribute, sublicense,
   and/or sell copies of the Software, and to permit persons to whom the
   Software is furnished to do so, subject to the following conditions:

   * Redistributions of source code must retain the above copyright notice,
     this list of conditions and the following disclaimers.

   * Redistributions in binary form must reproduce the above copyright
     notice, this list of conditions and the following disclaimers in the
     documentation and/or other materials provided with the distribution.

   * Neither the names of the LLVM Team, University of Illinois at
     Urbana-Champaign, nor the names of its contributors may be used to endorse
     or promote products derived from this Software without specific prior
     written permission.

   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
   IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
   CONTRIBUTORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
   OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS WITH
   THE SOFTWARE.

   This lgamma routine began with Sun's lgamma code from netlib.
   Their original copyright notice follows.

   ====================================================
   Copyright (C) 1993 by Sun Microsystems, Inc. All rights reserved.

   Developed at SunSoft, a Sun Microsystems, Inc. business.
   Permission to use, copy, modify, and distribute this
   software is freely granted, provided that this notice
   is preserved.
   ====================================================
*/

#include "templates.h"

static float
lgamma_absf (float ax)
{
  const float a0 = 7.72156649015328655494e-02f;
  const float a1 = 3.22467033424113591611e-01f;
  const float a2 = 6.73523010531292681824e-02f;
  const float a3 = 2.05808084325167332806e-02f;
  const float a4 = 7.38555086081402883957e-03f;
  const float a5 = 2.89051383673415629091e-03f;
  const float a6 = 1.19270763183362067845e-03f;
  const float a7 = 5.10069792153511336608e-04f;
  const float a8 = 2.20862790713908385557e-04f;
  const float a9 = 1.08011567247583939954e-04f;
  const float a10 = 2.52144565451257326939e-05f;
  const float a11 = 4.48640949618915160150e-05f;
  const float tc = 1.46163214496836224576e+00f;
  const float tf = -1.21486290535849611461e-01f;
  const float tt = -3.63867699703950536541e-18f;
  const float t0 = 4.83836122723810047042e-01f;
  const float t1 = -1.47587722994593911752e-01f;
  const float t2 = 6.46249402391333854778e-02f;
  const float t3 = -3.27885410759859649565e-02f;
  const float t4 = 1.79706750811820387126e-02f;
  const float t5 = -1.03142241298341437450e-02f;
  const float t6 = 6.10053870246291332635e-03f;
  const float t7 = -3.68452016781138256760e-03f;
  const float t8 = 2.25964780900612472250e-03f;
  const float t9 = -1.40346469989232843813e-03f;
  const float t10 = 8.81081882437654011382e-04f;
  const float t11 = -5.38595305356740546715e-04f;
  const float t12 = 3.15632070903625950361e-04f;
  const float t13 = -3.12754168375120860518e-04f;
  const float t14 = 3.35529192635519073543e-04f;
  const float u0 = -7.72156649015328655494e-02f;
  const float u1 = 6.32827064025093366517e-01f;
  const float u2 = 1.45492250137234768737e+00f;
  const float u3 = 9.77717527963372745603e-01f;
  const float u4 = 2.28963728064692451092e-01f;
  const float u5 = 1.33810918536787660377e-02f;
  const float v1 = 2.45597793713041134822e+00f;
  const float v2 = 2.12848976379893395361e+00f;
  const float v3 = 7.69285150456672783825e-01f;
  const float v4 = 1.04222645593369134254e-01f;
  const float v5 = 3.21709242282423911810e-03f;
  const float s0 = -7.72156649015328655494e-02f;
  const float s1 = 2.14982415960608852501e-01f;
  const float s2 = 3.25778796408930981787e-01f;
  const float s3 = 1.46350472652464452805e-01f;
  const float s4 = 2.66422703033638609560e-02f;
  const float s5 = 1.84028451407337715652e-03f;
  const float s6 = 3.19475326584100867617e-05f;
  const float r1 = 1.39200533467621045958e+00f;
  const float r2 = 7.21935547567138069525e-01f;
  const float r3 = 1.71933865632803078993e-01f;
  const float r4 = 1.86459191715652901344e-02f;
  const float r5 = 7.77942496381893596434e-04f;
  const float r6 = 7.32668430744625636189e-06f;
  const float w0 = 4.18938533204672725052e-01f;
  const float w1 = 8.33333333333329678849e-02f;
  const float w2 = -2.77777777728775536470e-03f;
  const float w3 = 7.93650558643019558500e-04f;
  const float w4 = -5.95187557450339963135e-04f;
  const float w5 = 8.36339918996282139126e-04f;
  const float w6 = -1.63092934096575273989e-03f;
  const float z1 = -0x1.2788d0p-1f;
  const float z2 = 0x1.a51a66p-1f;
  const float z3 = -0x1.9a4d56p-2f;
  const float z4 = 0x1.151322p-2f;

  uint uax = as_uint (ax);
  float ret;

  if (ax < 0x1.0p-6f)
    ret = fma (ax, fma (ax, fma (ax, fma (ax, z4, z3), z2), z1), -log (ax));
  else if (ax < 2.0f)
    {
      int i;
      int c;
      float y, t;
      if (ax <= 0.9f)
        {
          ret = -log (ax);
          y = 1.0f - ax;
          i = 0;
          c = ax < 0.7316f;
          t = ax - (tc - 1.0f);
          y = c ? t : y;
          i = c ? 1 : i;
          c = ax < 0.23164f;
          y = c ? ax : y;
          i = c ? 2 : i;
        }
      else
        {
          ret = 0.0f;
          y = 2.0f - ax;
          i = 0;
          c = ax < 1.7316f;
          t = ax - tc;
          y = c ? t : y;
          i = c ? 1 : i;
          c = ax < 1.23f;
          t = ax - 1.0f;
          y = c ? t : y;
          i = c ? 2 : i;
        }

      float z, w, p1, p2, p3, p;
      switch (i)
        {
        case 0:
          z = y * y;
          p1 = fma (z, fma (z, fma (z, fma (z, fma (z, a10, a8), a6), a4), a2),
                    a0);
          p2 = z * fma (z, fma (z, fma (z, fma (z, fma (z, a11, a9), a7), a5),
                                a3),
                        a1);
          p = fma (y, p1, p2);
          ret += fma (y, -0.5f, p);
          break;
        case 1:
          z = y * y;
          w = z * y;
          p1 = fma (w, fma (w, fma (w, fma (w, t12, t9), t6), t3), t0);
          p2 = fma (w, fma (w, fma (w, fma (w, t13, t10), t7), t4), t1);
          p3 = fma (w, fma (w, fma (w, fma (w, t14, t11), t8), t5), t2);
          p = fma (z, p1, -fma (w, -fma (y, p3, p2), tt));
          ret += tf + p;
          break;
        default:
          p1 = y * fma (y, fma (y, fma (y, fma (y, fma (y, u5, u4), u3), u2),
                                u1),
                        u0);
          p2 = fma (y, fma (y, fma (y, fma (y, fma (y, v5, v4), v3), v2), v1),
                    1.0f);
          ret += fma (y, -0.5f, p1 / p2);
          break;
        }
    }
  else if (ax < 8.0f)
    {
      int i = (int)ax;
      float y = ax - (float)i;
      float p = y * fma (y, fma (y, fma (y, fma (y, fma (y, fma (y, s6, s5),
                                                         s4),
                                                 s3),
                                         s2),
                                 s1),
                         s0);
      float q = fma (y, fma (y, fma (y, fma (y, fma (y, fma (y, r6, r5), r4),
                                            r3),
                                     r2),
                             r1),
                     1.0f);
      ret = fma (y, 0.5f, p / q);
      float z = 1.0f;
      z *= i > 2 ? y + 2.0f : 1.0f;
      z *= i > 3 ? y + 3.0f : 1.0f;
      z *= i > 4 ? y + 4.0f : 1.0f;
      z *= i > 5 ? y + 5.0f : 1.0f;
      z *= i > 6 ? y + 6.0f : 1.0f;
      ret += log (z);
    }
  else if (uax < 0x5c800000)
    {
      float z = 1.0f / ax;
      float y = z * z;
      float w = fma (z, fma (y, fma (y, fma (y, fma (y, fma (y, w6, w5), w4),
                                             w3),
                                     w2),
                             w1),
                     w0);
      ret = fma (ax - 0.5f, log (ax) - 1.0f, w);
    }
  else
    ret = fma (ax, log (ax), -ax);

  return ret;
}

static double
lgamma_absd (double ax)
{
  const double a0 = 7.72156649015328655494e-02;
  const double a1 = 3.22467033424113591611e-01;
  const double a2 = 6.73523010531292681824e-02;
  const double a3 = 2.05808084325167332806e-02;
  const double a4 = 7.38555086081402883957e-03;
  const double a5 = 2.89051383673415629091e-03;
  const double a6 = 1.19270763183362067845e-03;
  const double a7 = 5.10069792153511336608e-04;
  const double a8 = 2.20862790713908385557e-04;
  const double a9 = 1.08011567247583939954e-04;
  const double a10 = 2.52144565451257326939e-05;
  const double a11 = 4.48640949618915160150e-05;
  const double tc = 1.46163214496836224576e+00;
  const double tf = -1.21486290535849611461e-01;
  const double tt = -3.63867699703950536541e-18;
  const double t0 = 4.83836122723810047042e-01;
  const double t1 = -1.47587722994593911752e-01;
  const double t2 = 6.46249402391333854778e-02;
  const double t3 = -3.27885410759859649565e-02;
  const double t4 = 1.79706750811820387126e-02;
  const double t5 = -1.03142241298341437450e-02;
  const double t6 = 6.10053870246291332635e-03;
  const double t7 = -3.68452016781138256760e-03;
  const double t8 = 2.25964780900612472250e-03;
  const double t9 = -1.40346469989232843813e-03;
  const double t10 = 8.81081882437654011382e-04;
  const double t11 = -5.38595305356740546715e-04;
  const double t12 = 3.15632070903625950361e-04;
  const double t13 = -3.12754168375120860518e-04;
  const double t14 = 3.35529192635519073543e-04;
  const double u0 = -7.72156649015328655494e-02;
  const double u1 = 6.32827064025093366517e-01;
  const double u2 = 1.45492250137234768737e+00;
  const double u3 = 9.77717527963372745603e-01;
  const double u4 = 2.28963728064692451092e-01;
  const double u5 = 1.33810918536787660377e-02;
  const double v1 = 2.45597793713041134822e+00;
  const double v2 = 2.12848976379893395361e+00;
  const double v3 = 7.69285150456672783825e-01;
  const double v4 = 1.04222645593369134254e-01;
  const double v5 = 3.21709242282423911810e-03;
  const double s0 = -7.72156649015328655494e-02;
  const double s1 = 2.14982415960608852501e-01;
  const double s2 = 3.25778796408930981787e-01;
  const double s3 = 1.46350472652464452805e-01;
  const double s4 = 2.66422703033638609560e-02;
  const double s5 = 1.84028451407337715652e-03;
  const double s6 = 3.19475326584100867617e-05;
  const double r1 = 1.39200533467621045958e+00;
  const double r2 = 7.21935547567138069525e-01;
  const double r3 = 1.71933865632803078993e-01;
  const double r4 = 1.86459191715652901344e-02;
  const double r5 = 7.77942496381893596434e-04;
  const double r6 = 7.32668430744625636189e-06;
  const double w0 = 4.18938533204672725052e-01;
  const double w1 = 8.33333333333329678849e-02;
  const double w2 = -2.77777777728775536470e-03;
  const double w3 = 7.93650558643019558500e-04;
  const double w4 = -5.95187557450339963135e-04;
  const double w5 = 8.36339918996282139126e-04;
  const double w6 = -1.63092934096575273989e-03;
  const double z1 = -0x1.2788cfc6fb619p-1;
  const double z2 = 0x1.a51a6625307d3p-1;
  const double z3 = -0x1.9a4d55beab2d7p-2;
  const double z4 = 0x1.151322ac7d848p-2;
  const double z5 = -0x1.a8b9c17aa6149p-3;

  double ret;
  if (ax < 0x1.0p-8)
    ret = fma (ax, fma (ax, fma (ax, fma (ax, fma (ax, z5, z4), z3), z2), z1),
               -log (ax));
  else if (ax < 2.0)
    {
      int i;
      int c;
      double y, t;
      if (ax <= 0.9)
        {
          ret = -log (ax);
          y = 1.0 - ax;
          i = 0;
          c = ax < 0.7316;
          t = ax - (tc - 1.0);
          y = c ? t : y;
          i = c ? 1 : i;
          c = ax < 0.2316;
          y = c ? ax : y;
          i = c ? 2 : i;
        }
      else
        {
          ret = 0.0;
          y = 2.0 - ax;
          i = 0;
          c = ax < 1.7316;
          t = ax - tc;
          y = c ? t : y;
          i = c ? 1 : i;
          c = ax < 1.2316;
          t = ax - 1.0;
          y = c ? t : y;
          i = c ? 2 : i;
        }
      double w, z, p, p1, p2, p3;
      switch (i)
        {
        case 0:
          z = y * y;
          p1 = fma (z, fma (z, fma (z, fma (z, fma (z, a10, a8), a6), a4), a2),
                    a0);
          p2 = z * fma (z, fma (z, fma (z, fma (z, fma (z, a11, a9), a7), a5),
                                a3),
                        a1);
          p = fma (y, p1, p2);
          ret += fma (y, -0.5, p);
          break;
        case 1:
          z = y * y;
          w = z * y;
          p1 = fma (w, fma (w, fma (w, fma (w, t12, t9), t6), t3), t0);
          p2 = fma (w, fma (w, fma (w, fma (w, t13, t10), t7), t4), t1);
          p3 = fma (w, fma (w, fma (w, fma (w, t14, t11), t8), t5), t2);
          p = fma (z, p1, -fma (w, -fma (y, p3, p2), tt));
          ret += tf + p;
          break;
        default:
          p1 = y * fma (y, fma (y, fma (y, fma (y, fma (y, u5, u4), u3), u2),
                                u1),
                        u0);
          p2 = fma (y, fma (y, fma (y, fma (y, fma (y, v5, v4), v3), v2), v1),
                    1.0);
          ret += fma (y, -0.5, p1 / p2);
          break;
        }
    }
  else if (ax < 8.0)
    {
      int i = (int)ax;
      double y = ax - (double)i;
      double p = y * fma (y, fma (y, fma (y, fma (y, fma (y, fma (y, s6, s5),
                                                          s4),
                                                  s3),
                                          s2),
                                  s1),
                          s0);
      double q = fma (y, fma (y, fma (y, fma (y, fma (y, fma (y, r6, r5), r4),
                                             r3),
                                      r2),
                              r1),
                      1.0);
      ret = fma (y, 0.5, p / q);
      double z = 1.0;
      z *= i > 2 ? y + 2.0 : 1.0;
      z *= i > 3 ? y + 3.0 : 1.0;
      z *= i > 4 ? y + 4.0 : 1.0;
      z *= i > 5 ? y + 5.0 : 1.0;
      z *= i > 6 ? y + 6.0 : 1.0;
      ret += log (z);
    }
  else if (ax < 0x1.0p+58)
    {
      double z = 1.0 / ax;
      double y = z * z;
      double w = fma (z, fma (y, fma (y, fma (y, fma (y, fma (y, w6, w5), w4),
                                              w3),
                                      w2),
                              w1),
                      w0);
      ret = fma (ax - 0.5, log (ax) - 1.0, w);
    }
  else
    ret = fma (ax, log (ax), -ax);

  return ret;
}

float _CL_OVERLOADABLE
lgamma_r (float a, int __private *c)
{
  if (isnan (a))
    {
      *c = 0;
      return a;
    }
  if (isinf (a) || a == 0.0f)
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

  float ax = fabs (a);
  float ret = lgamma_absf (ax);
  if (a == 1.0f || a == 2.0f)
    return 0.0f;
  if (a > 0.0f)
    return ret;
  float t = sin (a * M_PI_F);
  return log (M_PI_F / fabs (t * a)) - ret;
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
  if (isinf (a) || a == 0.0)
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

  double ax = fabs (a);
  double ret = lgamma_absd (ax);
  if (a == 1.0 || a == 2.0)
    return 0.0;
  if (a > 0.0)
    return ret;
  double t = sin (a * M_PI);
  return log (M_PI / fabs (t * a)) - ret;
}

IMPLEMENT_BUILTIN_V_VPJ_ADDRSPACE (lgamma_r, double, int, __local)
IMPLEMENT_BUILTIN_V_VPJ_ADDRSPACE (lgamma_r, double, int, __global)
IF_GEN_AS (IMPLEMENT_BUILTIN_V_VPJ_ADDRSPACE (lgamma_r, double, int, __generic))
IMPLEMENT_BUILTIN_V_VPJ (lgamma_r, double2, int2, int, int, lo, hi)
IMPLEMENT_BUILTIN_V_VPJ (lgamma_r, double3, int3, int2, int, lo, s2)
IMPLEMENT_BUILTIN_V_VPJ (lgamma_r, double4, int4, int2, int2, lo, hi)
IMPLEMENT_BUILTIN_V_VPJ (lgamma_r, double8, int8, int4, int4, lo, hi)
IMPLEMENT_BUILTIN_V_VPJ (lgamma_r, double16, int16, int8, int8, lo, hi))
