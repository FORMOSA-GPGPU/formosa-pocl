/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "../templates.h"

_CL_OVERLOADABLE _CL_READNONE float
erf(float x)
{
    float ax = fabs(x);
    float ret;

    if (ax < 1.0f) {
        float t = ax*ax;
        ret = fma(t, fma(t, fma(t, fma(t,
              fma(t, fma(t,
                  0x1.496a32p-14f, -0x1.a3f700p-11f), 0x1.5405b2p-8f), -0x1.b7f90ep-6f),
                  0x1.ce2cf8p-4f), -0x1.81273ep-2f), 0x1.20dd74p+0f),
        ret = ax * ret;
    } else if (ax < 1.75f) {
        float t = ax - 1.0f;
        ret = fma(t, fma(t, fma(t, fma(t,
              fma(t, fma(t,
                  0x1.56793ap-6f, -0x1.3f1e8ap-4f), 0x1.254d82p-4f), 0x1.1abe80p-3f),
                  -0x1.a90f70p-2f), 0x1.a91224p-2f), 0x1.af767ap-1f);

    } else if (ax < 2.5f) {
        float t = ax - 1.75f;
        ret = fma(t, fma(t, fma(t, fma(t,
              fma(t, fma(t,
                  -0x1.6c7276p-10f, 0x1.f0807ep-7f), -0x1.a90488p-5f), 0x1.74f388p-4f),
                  -0x1.7ab6aap-4f), 0x1.b05ea0p-5f), 0x1.f92d06p-1f);
    } else if (ax < 3.9375f) {
        float t = ax - 2.5f;
        ret = fma(t, fma(t, fma(t, fma(t,
              fma(t, fma(t,
                  0x1.b94482p-16f, -0x1.9d0710p-13f), -0x1.b43cb6p-12f), 0x1.01db88p-7f),
                  -0x1.e85a32p-10f), -0x1.a49baep-3f), 0x1.3a50e2p-1f);
        ret = ret * ret;
        ret = ret * ret;
        ret = ret * ret;
        ret = fma(-ret, ret, 1.0f);
    } else {
        ret = isnan(x) ? x : 1.0f;
    }

    ret = copysign(ret, x);
    return ret;
}


IMPLEMENT_BUILTIN_V_V (erf, float2, NAME1_2 (erf))
IMPLEMENT_BUILTIN_V_V (erf, float3, NAME1_3 (erf))
IMPLEMENT_BUILTIN_V_V (erf, float4, NAME1_4 (erf))
IMPLEMENT_BUILTIN_V_V (erf, float8, NAME1_8 (erf))
IMPLEMENT_BUILTIN_V_V (erf, float16, NAME1_16 (erf))
