/*
 * Standalone INDI golden-reference generator
 *
 * Extracts the core INDI math from indiflight (indi.c / indi_init.c) and runs
 * it with controlled inputs, producing CSV golden vectors on stdout.
 *
 * This file is self-contained: it reimplements or stubs every indiflight
 * dependency so that the real ActiveSetCtlAlloc library can be linked in
 * without pulling the full firmware.
 *
 * Build with the accompanying Makefile.
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <stdint.h>
#include <math.h>
#include <float.h>

/* ------------------------------------------------------------------ */
/*  Platform / firmware macros we need                                 */
/* ------------------------------------------------------------------ */

#define USE_INDI
#define USE_ACC

#ifndef AS_SINGLE_FLOAT
#define AS_SINGLE_FLOAT
#endif

/* Sizes — must match before including ActiveSetCtlAlloc headers */
#define MAX_SUPPORTED_MOTORS 4
#define MAXU  MAX_SUPPORTED_MOTORS
#define MAX_SUPPORTED_PSEUDOCONTROLS 6
#define MAXV  MAX_SUPPORTED_PSEUDOCONTROLS

#ifndef AS_N_U
#define AS_N_U  MAXU
#endif
#ifndef AS_N_V
#define AS_N_V  MAXV
#endif

/* Firmware compatibility shims */
#define FAST_CODE
#define FAST_CODE_NOINLINE
#define FAST_DATA_ZERO_INIT
#define FAST_DATA
#define UNUSED(x) (void)(x)

#define XYZ_AXIS_COUNT 3

typedef enum { FD_ROLL = 0, FD_PITCH, FD_YAW } flight_dynamics_index_t;
typedef enum { X = 0, Y, Z } axis_e;

/* ------------------------------------------------------------------ */
/*  Maths types & helpers  (from common/maths.h)                      */
/* ------------------------------------------------------------------ */

#ifndef sq
#define sq(x) ((x)*(x))
#endif

#define M_PIf       3.14159265358979323846f
#define GRAVITYf    9.80665f
#define DEGREES_TO_RADIANS(angle) ((angle) * 0.0174532925f)
#define RADIANS_TO_DEGREES(angle) ((angle) * 57.2957796f)
#define SECONDS_PER_MINUTE  60.0f
#define ERPM_PER_LSB        100.0f

#define MIN(a,b) \
  __extension__ ({ __typeof__ (a) _a = (a); \
  __typeof__ (b) _b = (b); \
  _a < _b ? _a : _b; })
#define MAX(a,b) \
  __extension__ ({ __typeof__ (a) _a = (a); \
  __typeof__ (b) _b = (b); \
  _a > _b ? _a : _b; })

static inline float constrainf(float amt, float low, float high) {
    if (amt < low)  return low;
    if (amt > high) return high;
    return amt;
}

static inline unsigned int constrainu(unsigned int amt, unsigned int low,
                                      unsigned int high) {
    if (amt < low)  return low;
    if (amt > high) return high;
    return amt;
}

/* -- Trig approximations (matching indiflight VERY_FAST_MATH) -- */

#define sinPolyCoef3 -1.666568107e-1f
#define sinPolyCoef5  8.312366210e-3f
#define sinPolyCoef7 -1.849218155e-4f
#define sinPolyCoef9  0

float sin_approx(float x) {
    int32_t xint = (int32_t)x;
    if (xint < -32 || xint > 32) return sinf(x);
    while (x >  M_PIf) x -= (2.0f * M_PIf);
    while (x < -M_PIf) x += (2.0f * M_PIf);
    if (x >  (0.5f * M_PIf)) x =  (0.5f * M_PIf) - (x - (0.5f * M_PIf));
    else if (x < -(0.5f * M_PIf)) x = -(0.5f * M_PIf) - ((0.5f * M_PIf) + x);
    float x2 = x * x;
    return x + x * x2 * (sinPolyCoef3 + x2 * (sinPolyCoef5 + x2 * (sinPolyCoef7 + x2 * sinPolyCoef9)));
}

float cos_approx(float x) {
    return sin_approx(x + (0.5f * M_PIf));
}

float acos_approx(float x) {
    float xa = fabsf(x);
    float result = sqrtf(1.0f - xa) * (1.5707288f + xa * (-0.2121144f + xa * (0.0742610f + (-0.0187293f * xa))));
    if (x < 0.0f) return M_PIf - result;
    return result;
}

/* -- fp_vector / fp_quaternion types -- */

typedef struct fp_vector { float X,Y,Z; } fp_vector_def;
typedef union u_fp_vector { float A[3]; fp_vector_def V; } fp_vector_t;

typedef struct fp_quaternion { float w,x,y,z; } fp_quaternion_t;

#define VEC3_SCALAR_MULT_ADD(_orig, _sc, _add) { \
    _orig.V.X += _sc * _add.V.X; \
    _orig.V.Y += _sc * _add.V.Y; \
    _orig.V.Z += _sc * _add.V.Z; \
}

#define VEC3_SCALAR_MULT(_orig, _sc) { \
    _orig.V.X *= _sc; \
    _orig.V.Y *= _sc; \
    _orig.V.Z *= _sc; \
}

#define VEC3_XY_LENGTH(_orig) \
    sqrtf(_orig.V.X*_orig.V.X + _orig.V.Y*_orig.V.Y)

#define VEC3_CONSTRAIN_XY_LENGTH(_vec, _max_length) { \
    float _vec_len = VEC3_XY_LENGTH(_vec); \
    _vec.V.X /= constrainf(_vec_len / _max_length, 1.f, +FLT_MAX); \
    _vec.V.Y /= constrainf(_vec_len / _max_length, 1.f, +FLT_MAX); \
}

#define QUAT_SCALAR_MULT(_orig, _sc) { \
    _orig.w *= _sc; _orig.x *= _sc; _orig.y *= _sc; _orig.z *= _sc; \
}

/* We need chain_quaternion and quatRotMatCol for getAlphaSpBody
   but we bypass attitude control in our tests (controlAttitude=false),
   so we only need stub-level versions. Provide real ones anyway for
   completeness in case tests are extended. */

fp_quaternion_t chain_quaternion(const fp_quaternion_t* qA_I, const fp_quaternion_t* qB_A) {
    fp_quaternion_t out = {
        .w = qA_I->w * qB_A->w - qA_I->x * qB_A->x - qA_I->y * qB_A->y - qA_I->z * qB_A->z,
        .x = qA_I->x * qB_A->w + qA_I->w * qB_A->x + qA_I->y * qB_A->z - qA_I->z * qB_A->y,
        .y = qA_I->w * qB_A->y - qA_I->x * qB_A->z + qA_I->y * qB_A->w + qA_I->z * qB_A->x,
        .z = qA_I->w * qB_A->z + qA_I->x * qB_A->y - qA_I->y * qB_A->x + qA_I->z * qB_A->w
    };
    return out;
}

fp_vector_t quatRotMatCol(const fp_quaternion_t* q, uint8_t axis) {
    fp_vector_t res = {0};
    switch(axis) {
        case 0:
            res.V.X = 1 - 2*(q->y*q->y + q->z*q->z);
            res.V.Y = 2*q->x*q->y + 2*q->w*q->z;
            res.V.Z = 2*q->x*q->z - 2*q->w*q->y;
            break;
        case 1:
            res.V.X = 2*q->x*q->y - 2*q->w*q->z;
            res.V.Y = 1 - 2*(q->x*q->x + q->z*q->z);
            res.V.Z = 2*q->y*q->z + 2*q->w*q->x;
            break;
        case 2:
            res.V.X = 2*q->x*q->z + 2*q->w*q->y;
            res.V.Y = 2*q->y*q->z - 2*q->w*q->x;
            res.V.Z = 1 - 2*(q->x*q->x + q->y*q->y);
            break;
    }
    return res;
}

void quaternion_of_axis_angle(fp_quaternion_t *q, const fp_vector_t *ax, float angle) {
    float ang2 = angle * 0.5f;
    float cang2 = cos_approx(ang2);
    float sang2 = sin_approx(ang2);
    q->w = cang2;
    q->x = ax->V.X * sang2;
    q->y = ax->V.Y * sang2;
    q->z = ax->V.Z * sang2;
}

/* ------------------------------------------------------------------ */
/*  Filter implementations  (from common/filter.c)                    */
/* ------------------------------------------------------------------ */

typedef struct pt1Filter_s { float state; float k; } pt1Filter_t;

typedef struct biquadFilter_s {
    float b0, b1, b2, a1, a2;
    float x1, x2, y1, y2;
    float weight;
} biquadFilter_t;

/* Filter function pointer type (used in gyro struct but we don't need it) */
struct filter_s;
typedef struct filter_s filter_t;
typedef float (*filterApplyFnPtr)(filter_t *filter, float input);

float pt1FilterGain(float f_cut, float dT) {
    float RC = 1.0f / (2.0f * M_PIf * f_cut);
    return dT / (RC + dT);
}

void pt1FilterInit(pt1Filter_t *filter, float k) {
    filter->state = 0.0f;
    filter->k = k;
}

float pt1FilterApply(pt1Filter_t *filter, float input) {
    filter->state = filter->state + filter->k * (input - filter->state);
    return filter->state;
}

#define BIQUAD_Q (1.0f / sqrtf(2.0f))

typedef enum { FILTER_LPF, FILTER_NOTCH, FILTER_BPF } biquadFilterType_e;

void biquadFilterUpdate(biquadFilter_t *filter, float filterFreq,
                        uint32_t refreshRate, float Q,
                        biquadFilterType_e filterType, float weight) {
    const float omega = 2.0f * M_PIf * filterFreq * refreshRate * 0.000001f;
    const float sn = sin_approx(omega);
    const float cs = cos_approx(omega);
    const float alpha = sn / (2.0f * Q);

    switch (filterType) {
    case FILTER_LPF:
        filter->b1 = 1 - cs;
        filter->b0 = filter->b1 * 0.5f;
        filter->b2 = filter->b0;
        filter->a1 = -2 * cs;
        filter->a2 = 1 - alpha;
        break;
    case FILTER_NOTCH:
        filter->b0 = 1;
        filter->b1 = -2 * cs;
        filter->b2 = 1;
        filter->a1 = filter->b1;
        filter->a2 = 1 - alpha;
        break;
    case FILTER_BPF:
        filter->b0 = alpha;
        filter->b1 = 0;
        filter->b2 = -alpha;
        filter->a1 = -2 * cs;
        filter->a2 = 1 - alpha;
        break;
    }

    const float a0 = 1 + alpha;
    filter->b0 /= a0;
    filter->b1 /= a0;
    filter->b2 /= a0;
    filter->a1 /= a0;
    filter->a2 /= a0;
    filter->weight = weight;
}

void biquadFilterInit(biquadFilter_t *filter, float filterFreq,
                      uint32_t refreshRate, float Q,
                      biquadFilterType_e filterType, float weight) {
    biquadFilterUpdate(filter, filterFreq, refreshRate, Q, filterType, weight);
    filter->x1 = filter->x2 = 0;
    filter->y1 = filter->y2 = 0;
}

void biquadFilterInitLPF(biquadFilter_t *filter, float filterFreq,
                         uint32_t refreshRate) {
    biquadFilterInit(filter, filterFreq, refreshRate, BIQUAD_Q, FILTER_LPF, 1.0f);
}

float biquadFilterApply(biquadFilter_t *filter, float input) {
    const float result = filter->b0 * input + filter->x1;
    filter->x1 = filter->b1 * input - filter->a1 * result + filter->x2;
    filter->x2 = filter->b2 * input - filter->a2 * result;
    return result;
}

/* ------------------------------------------------------------------ */
/*  Actuator linearization  (from indi.c)                             */
/* ------------------------------------------------------------------ */

typedef struct actLin_s { float A; float B; float C; float k; } actLin_t;

void updateLinearization(actLin_t* lin, float k) {
    lin->k = constrainf(k, 0.025f, 0.7f);
    lin->A = 1.f / lin->k;
    lin->B = (sq(lin->k) - 2.f*lin->k + 1.f) / (4.f*sq(lin->k));
    lin->C = (lin->k - 1) / (2.f*lin->k);
}

float indiLinearization(actLin_t* lin, float in) {
    if ((lin->A < 1.f) || (lin->B < 0.f)) return in;
    if ((in <= 0.f) || (in >= 1.f))        return in;
    return sqrtf(lin->A*in + lin->B) + lin->C;
}

float indiOutputCurve(actLin_t* lin, float in) {
    return lin->k*sq(in) + (1-lin->k)*in;
}

/* ------------------------------------------------------------------ */
/*  WLS / ActiveSet solver — include from library                      */
/* ------------------------------------------------------------------ */

/* Already defined: AS_N_U, AS_N_V, AS_SINGLE_FLOAT */
#include "solveActiveSet.h"
#include "setupWLS.h"

/* ------------------------------------------------------------------ */
/*  indiRuntime_t — faithful copy from indi.h                         */
/* ------------------------------------------------------------------ */

typedef struct indiRuntime_s {
    /* Att/Rate config */
    fp_vector_t attGains;
    fp_vector_t rateGains;
    float attMaxTiltRate;
    float attMaxYawRate;
    uint8_t attRateDenom;
    bool manualUseCoordinatedYaw;
    float manualMaxUpwardsSpf;
    float manualMaxTilt;
    /* general INDI config */
    bool useIncrement;
    bool useConstantG2;
    bool useRpmFeedback;
    bool useRpmDotFeedback;
    fp_vector_t maxRateSp;
    /* actuator config */
    uint8_t actNum;
    float actMaxOmega[MAXU];
    float actMaxOmega2[MAXU];
    float actHoverOmega[MAXU];
    float actTimeConstS[MAXU];
    float actNonlinearity[MAXU];
    float actLimit[MAXU];
    float actG1[MAXV][MAXU];
    float actG2[3][MAXU];
    float G2_scaler[MAXU];
    /* Filtering */
    float imuSyncLp2Hz;
    /* WLS config */
    float wlsWv[MAXV];
    float wlsWu[MAXU];
    float u_pref[MAXU];
    activeSetAlgoChoice wlsAlgo;
    bool useWls;
    bool wlsWarmstart;
    uint8_t wlsMaxIter;
    float wlsCondBound;
    float wlsTheta;
    uint8_t wlsNanLimit;
    /* runtime — actuators */
    float d[MAXU];
    float u[MAXU];
    float uState[MAXU];
    float uState_fs[MAXU];
    actLin_t lin[MAXU];
    float omega[MAXU];
    float omega_fs[MAXU];
    float omegaDot_fs[MAXU];
    float erpmToRads;
    /* runtime — axes */
    fp_vector_t attGainsCasc;
    fp_quaternion_t attSpNed;
    fp_quaternion_t attErrBody;
    fp_vector_t rateSpBody;
    fp_vector_t rateSpBodyCommanded;
    fp_vector_t rateDotSpBody;
    fp_vector_t spfSpBody;
    float dv[MAXV];
    fp_vector_t rate;
    fp_vector_t rateDot;
    fp_vector_t rateDot_fs;
    fp_vector_t spf;
    fp_vector_t spf_fs;
    /* filters */
    pt1Filter_t uLagFilter[MAXU];
    biquadFilter_t uStateFilter[MAXU];
    biquadFilter_t omegaFilter[MAXU];
    biquadFilter_t rateFilter[3];
    biquadFilter_t spfFilter[3];
    /* housekeeping */
    float dT;
    float indiFrequency;
    uint8_t attExecCounter;
    uint16_t nanCounter;
    /* control law selection */
    bool bypassControl;
    bool controlAttitude;
    bool trackAttitudeYaw;
} indiRuntime_t;

indiRuntime_t indiRun;

/* ------------------------------------------------------------------ */
/*  Stubbed globals that indi.c reads                                  */
/* ------------------------------------------------------------------ */

typedef struct {
    struct { float acc_1G_rec; } dev;
} stub_acc_t;

stub_acc_t acc;

/* gyro — only the fields indi.c reads */
typedef struct {
    float gyroADCafterRpm[3];
    float gyroADCf[3];
    uint32_t targetLooptime;  /* microseconds */
} stub_gyro_t;

stub_gyro_t gyro;

/* acc raw data in ADC counts — indi.c computes  accADCafterRpm * acc_1G_rec * g */
float acc_accADCafterRpm[3];

/* arming / ground flags — controlled per test */
typedef uint32_t armingFlag_e;
armingFlag_e armingFlags;
#define ARMED  (1 << 0)
#define ARMING_FLAG(mask)        (armingFlags & (mask))
#define ENABLE_ARMING_FLAG(mask) (armingFlags |= (mask))

static bool stub_touchingGround = false;
bool isTouchingGround(void) { return stub_touchingGround; }

void disarm(int reason) { (void)reason; /* no-op in test */ }

/* ------------------------------------------------------------------ */
/*  getMotorCommands — rewritten from indi.c with stubs plugged in    */
/* ------------------------------------------------------------------ */

/*
 * We keep this as close to the original as possible.
 * Static locals are replicated; they persist across calls within a test,
 * but we add a reset helper to clear them between test cases.
 */

static float gmc_du[MAXU];
static float gmc_rate_prev[XYZ_AXIS_COUNT];
static int8_t  gmc_Ws[MAXU];
static activeSetExitCode gmc_as_exit_code;

static void reset_getMotorCommands_state(void) {
    memset(gmc_du, 0, sizeof(gmc_du));
    memset(gmc_rate_prev, 0, sizeof(gmc_rate_prev));
    memset(gmc_Ws, 0, sizeof(gmc_Ws));
    gmc_as_exit_code = 0; /* AS_SUCCESS */
}

void getMotorCommands(void) {
    /* Sensor read-in & filtering (lines 369-384 of indi.c) */
    for (int axis = FD_ROLL; axis <= FD_YAW; axis++) {
        indiRun.rate.A[axis] = DEGREES_TO_RADIANS(gyro.gyroADCafterRpm[axis]);
        indiRun.spf.A[axis]  = acc_accADCafterRpm[axis] * acc.dev.acc_1G_rec * GRAVITYf;

        indiRun.rateDot.A[axis] = indiRun.indiFrequency * (indiRun.rate.A[axis] - gmc_rate_prev[axis]);
        gmc_rate_prev[axis] = indiRun.rate.A[axis];

        indiRun.rateDot_fs.A[axis] = biquadFilterApply(&indiRun.rateFilter[axis], indiRun.rateDot.A[axis]);
        indiRun.spf_fs.A[axis]     = biquadFilterApply(&indiRun.spfFilter[axis],  indiRun.spf.A[axis]);
    }

    /* Omega inverse (lines 387-393) */
    float omega_inv[MAXU];
    for (int i = 0; i < indiRun.actNum; i++) {
        float invThresh = 0.1f * indiRun.actMaxOmega[i];
        omega_inv[i] = (fabsf(indiRun.omega_fs[i]) > invThresh)
                        ? 1.f / indiRun.omega_fs[i]
                        : 1.f / invThresh;
    }

    /* Motor acceleration fallback (lines 396-407) */
    for (int i = 0; i < indiRun.actNum; i++) {
        /* No dshot telemetry in test — always use fallback */
        indiRun.omegaDot_fs[i] = gmc_du[i] * indiRun.G2_scaler[i] * omega_inv[i];
    }

    /* bypassControl check */
    if (indiRun.bypassControl) return;

    /* doIndi flag (line 414) */
    bool doIndi = (!isTouchingGround()) && ARMING_FLAG(ARMED);

    /* Pseudocontrol (lines 417-429) */
    indiRun.dv[0] = 0.f;
    indiRun.dv[1] = 0.f;
    indiRun.dv[2] = indiRun.spfSpBody.V.Z - doIndi * indiRun.spf_fs.V.Z;
    indiRun.dv[3] = indiRun.rateDotSpBody.V.X - doIndi * indiRun.rateDot_fs.V.X;
    indiRun.dv[4] = indiRun.rateDotSpBody.V.Y - doIndi * indiRun.rateDot_fs.V.Y;
    indiRun.dv[5] = indiRun.rateDotSpBody.V.Z - doIndi * indiRun.rateDot_fs.V.Z;

    for (int j = 0; j < 3; j++) {
        for (int i = 0; i < indiRun.actNum; i++) {
            indiRun.dv[j+3] += doIndi * indiRun.actG2[j][i] * indiRun.omegaDot_fs[i];
        }
    }

    /* G1+G2 matrix (lines 433-439) */
    float G1G2[MAXU * MAXV];
    for (int i = 0; i < indiRun.actNum; i++) {
        for (int j = 0; j < MAXV; j++) {
            G1G2[MAXV*i + j] = indiRun.actG1[j][i];
            if (j > 2)
                G1G2[MAXV*i + j] += indiRun.G2_scaler[i] * omega_inv[i] * indiRun.actG2[j-3][i];
        }
    }

    /* WLS setup (lines 442-486) */
    float gamma_used;
    float A_as[(MAXU+MAXV) * MAXU];
    float b_as[(MAXU+MAXV)];
    float du_as[MAXU];
    float du_min[MAXU];
    float du_max[MAXU];
    float du_pref[MAXU];

    for (int i = 0; i < indiRun.actNum; i++) {
        du_min[i]  = 0.f - doIndi * indiRun.uState_fs[i];
        du_max[i]  = indiRun.actLimit[i] - doIndi * indiRun.uState_fs[i];
        du_pref[i] = 0.f - doIndi * indiRun.uState_fs[i];
    }

    float Wu_as[MAXU];
    for (int i = 0; i < indiRun.actNum; i++)
        Wu_as[i] = indiRun.wlsWu[i];

    setupWLS_A(G1G2, indiRun.wlsWv, Wu_as, MAXV, indiRun.actNum,
               indiRun.wlsTheta, indiRun.wlsCondBound, A_as, &gamma_used);
    setupWLS_b(indiRun.dv, du_pref, indiRun.wlsWv, Wu_as, MAXV,
               indiRun.actNum, gamma_used, b_as);

    for (int i = 0; i < indiRun.actNum; i++) {
        du_as[i] = (du_min[i] + du_max[i]) * 0.5f;
        if (gmc_as_exit_code >= AS_NAN_FOUND_Q)
            gmc_Ws[i] = 0;
    }

    int iterations;
    int n_free;
    float alloc_costs[1] = {0.f};

    gmc_as_exit_code = solveActiveSet(indiRun.wlsAlgo)(
        A_as, b_as, du_min, du_max, du_as, gmc_Ws,
        indiRun.wlsMaxIter, indiRun.actNum, MAXV,
        &iterations, &n_free, alloc_costs);

    if (gmc_as_exit_code >= AS_NAN_FOUND_Q) {
        indiRun.nanCounter++;
    } else {
        if (ARMING_FLAG(ARMED))
            indiRun.nanCounter = 0;
    }

    /* Apply allocation results (lines 502-515) */
    for (int i = 0; i < indiRun.actNum; i++) {
        indiRun.uState_fs[i] = biquadFilterApply(&indiRun.uStateFilter[i], indiRun.uState[i]);
        indiRun.uState_fs[i] = constrainf(indiRun.uState_fs[i], 0.f, 1.f);

        if (gmc_as_exit_code < AS_NAN_FOUND_Q)
            indiRun.u[i] = constrainf(doIndi*indiRun.uState_fs[i] + du_as[i],
                                      0.f, indiRun.actLimit[i]);

        gmc_du[i] = indiRun.u[i] - indiRun.uState[i];

        indiRun.d[i] = indiLinearization(&indiRun.lin[i], indiRun.u[i]);
    }
}

/* ------------------------------------------------------------------ */
/*  indiUpdateActuatorState — from indi.c line 521                    */
/* ------------------------------------------------------------------ */

void indiUpdateActuatorState(float* d_in) {
    for (int i = 0; i < indiRun.actNum; i++) {
        float u_val = indiOutputCurve(&indiRun.lin[i], d_in[i]);
        indiRun.uState[i] = pt1FilterApply(&indiRun.uLagFilter[i], u_val);

        /* No dshot telemetry — use fallback hover omega */
        indiRun.omega[i]    = indiRun.actHoverOmega[i];
        indiRun.omega_fs[i] = indiRun.actHoverOmega[i];
    }
}

/* ------------------------------------------------------------------ */
/*  getAlphaSpBody — rate error * gain  (simplified: no attitude ctl) */
/* ------------------------------------------------------------------ */

void getAlphaSpBody(void) {
    /* Rate estimation from gyro.gyroADCf (already-filtered gyro).
       In the test we set gyroADCf = gyroADCafterRpm for simplicity. */
    fp_vector_t rateEstBody = {
        .V.X = DEGREES_TO_RADIANS(gyro.gyroADCf[FD_ROLL]),
        .V.Y = DEGREES_TO_RADIANS(gyro.gyroADCf[FD_PITCH]),
        .V.Z = DEGREES_TO_RADIANS(gyro.gyroADCf[FD_YAW]),
    };

    /* rateErr = rateSpBody - rateEstBody */
    fp_vector_t rateErr = indiRun.rateSpBody;
    VEC3_SCALAR_MULT_ADD(rateErr, -1.0f, rateEstBody);

    /* rateDotSpBody = rateGains * rateErr */
    indiRun.rateDotSpBody.V.X = indiRun.rateGains.V.X * rateErr.V.X;
    indiRun.rateDotSpBody.V.Y = indiRun.rateGains.V.Y * rateErr.V.Y;
    indiRun.rateDotSpBody.V.Z = indiRun.rateGains.V.Z * rateErr.V.Z;
}

/* ------------------------------------------------------------------ */
/*  Init the INDI runtime — mirrors initIndiRuntime() from            */
/*  indi_init.c but with hardcoded parameters from the test spec      */
/* ------------------------------------------------------------------ */

/* Profile parameters — derived from cybflight QuadX physical geometry (NED frame).
 *
 * Motor layout (Betaflight QuadX):
 *   M0=RR(CW), M1=FR(CCW), M2=RL(CCW), M3=FL(CW)
 *
 * FLU positions: M0(-0.075,-0.1), M1(+0.075,-0.1), M2(-0.075,+0.1), M3(+0.075,+0.1)
 * NED positions (py_NED = -py_FLU): M0(-0.075,+0.1), M1(+0.075,+0.1), M2(-0.075,-0.1), M3(+0.075,-0.1)
 *
 * Body: mass=0.55 kg, Ixx=0.0025, Iyy=0.0021, Izz=0.0043 kg·m²
 * T=8.5 N, c=0.022 m
 *
 * NED G1 (acceleration space):
 *   fz    = -T/m = -15.4545 N/kg
 *   roll  = -py_NED * T / Ixx
 *   pitch = px * T / Iyy         (from τ_y = r_z*F_x - r_x*F_z = px*T)
 *   yaw   = -s * c * T / Izz     (CW = negative yaw in NED)
 *
 * Config scaling: fz×100, roll/pitch/yaw×10
 */
static const int16_t  cfg_actG1_fz[4]    = {-1545, -1545, -1545, -1545};
static const int16_t  cfg_actG1_roll[4]  = {-3400, -3400, 3400, 3400};
static const int16_t  cfg_actG1_pitch[4] = {-3036, 3036, -3036, 3036};
static const int16_t  cfg_actG1_yaw[4]   = {-435, 435, 435, -435};

#define CFG_ACT_NUM       4
#define CFG_ACT_TIME_MS   25
#define CFG_ACT_MAX_RPM   40000
#define CFG_ACT_HOVER_RPM 20000
#define CFG_ACT_NONLIN    50
#define CFG_ACT_LIMIT     100
#define CFG_USE_INCREMENT 1
#define CFG_IMU_SYNC_LP2  15
#define CFG_LOOP_US       125   /* 8 kHz */

static const uint16_t cfg_rateGains[3] = {200, 200, 200};
static const uint8_t  cfg_wlsWv[6]    = {1, 1, 50, 50, 50, 5};
static const uint8_t  cfg_wlsWu[4]    = {1, 1, 1, 1};

static void initIndiRuntime(void) {
    memset(&indiRun, 0, sizeof(indiRun));

    /* Rate gains (profile value * 0.1) */
    indiRun.rateGains.A[0] = MAX(1U, cfg_rateGains[0]) * 0.1f;
    indiRun.rateGains.A[1] = MAX(1U, cfg_rateGains[1]) * 0.1f;
    indiRun.rateGains.A[2] = MAX(1U, cfg_rateGains[2]) * 0.1f;

    indiRun.attMaxTiltRate = DEGREES_TO_RADIANS(800);
    indiRun.attMaxYawRate  = DEGREES_TO_RADIANS(400);
    indiRun.attRateDenom   = 4;

    /* max rate setpoint defaults */
    indiRun.maxRateSp.A[0] = DEGREES_TO_RADIANS(1800);
    indiRun.maxRateSp.A[1] = DEGREES_TO_RADIANS(1800);
    indiRun.maxRateSp.A[2] = DEGREES_TO_RADIANS(1800);

    indiRun.useIncrement     = (bool)CFG_USE_INCREMENT;
    indiRun.useConstantG2    = false;
    indiRun.useRpmFeedback   = false;
    indiRun.useRpmDotFeedback = false;

    indiRun.actNum = CFG_ACT_NUM;

    for (int i = 0; i < MAXU; i++) {
        float hoverRpm = (float)MAX(100U, (unsigned)CFG_ACT_HOVER_RPM);
        indiRun.actHoverOmega[i] = hoverRpm / SECONDS_PER_MINUTE * 2.f * M_PIf;

        float maxRpm = (float)MAX(100U, (unsigned)CFG_ACT_MAX_RPM);
        indiRun.actMaxOmega[i]  = maxRpm / SECONDS_PER_MINUTE * 2.f * M_PIf;
        indiRun.actMaxOmega2[i] = sq(indiRun.actMaxOmega[i]);

        indiRun.actTimeConstS[i] = MAX(1UL, (unsigned long)CFG_ACT_TIME_MS) * 1e-3f;
        indiRun.actNonlinearity[i] = constrainu(CFG_ACT_NONLIN, 0, 100) * 0.01f;
        indiRun.actLimit[i] = constrainu(CFG_ACT_LIMIT, 0, 100) * 0.01f;

        indiRun.actG1[0][i] = 0;   /* fx */
        indiRun.actG1[1][i] = 0;   /* fy */
        indiRun.actG1[2][i] = cfg_actG1_fz[i]    * 0.01f;
        indiRun.actG1[3][i] = cfg_actG1_roll[i]   * 0.1f;
        indiRun.actG1[4][i] = cfg_actG1_pitch[i]  * 0.1f;
        indiRun.actG1[5][i] = cfg_actG1_yaw[i]    * 0.1f;

        indiRun.actG2[0][i] = 0.f;
        indiRun.actG2[1][i] = 0.f;
        indiRun.actG2[2][i] = 0.f;

        indiRun.G2_scaler[i] = 0.5f * indiRun.actMaxOmega2[i] / indiRun.actTimeConstS[i];

        indiRun.wlsWu[i] = (float)cfg_wlsWu[i];
        indiRun.u_pref[i] = 0.f;

        /* Zero runtime state */
        indiRun.d[i]         = 0.f;
        indiRun.u[i]         = 0.f;
        indiRun.uState[i]    = 0.f;
        indiRun.uState_fs[i] = 0.f;
        updateLinearization(&indiRun.lin[i], indiRun.actNonlinearity[i]);
        indiRun.omega[i]     = 0.f;
        indiRun.omega_fs[i]  = 0.f;
        indiRun.omegaDot_fs[i] = 0.f;
    }

    for (int i = 0; i < MAXV; i++)
        indiRun.wlsWv[i] = (float)cfg_wlsWv[i];

    indiRun.imuSyncLp2Hz = (float)CFG_IMU_SYNC_LP2;

    indiRun.wlsAlgo      = (activeSetAlgoChoice)1; /* AS_QR */
    indiRun.useWls        = true;
    indiRun.wlsWarmstart  = true;
    indiRun.wlsMaxIter    = 1;
    indiRun.wlsCondBound  = ((uint16_t)(1 << 15)) * 1e4f;
    indiRun.wlsTheta      = 1 * 1e-4f;
    indiRun.wlsNanLimit   = 20;

    /* Axes */
    for (int axis = FD_ROLL; axis <= FD_YAW; axis++) {
        indiRun.attGainsCasc.A[axis] = 0.f;
        indiRun.rateSpBody.A[axis]    = 0.f;
        indiRun.rateDotSpBody.A[axis] = 0.f;
        indiRun.spfSpBody.A[axis]     = 0.f;
        indiRun.rate.A[axis]          = 0.f;
        indiRun.rateDot.A[axis]       = 0.f;
        indiRun.rateDot_fs.A[axis]    = 0.f;
        indiRun.spf.A[axis]           = 0.f;
        indiRun.spf_fs.A[axis]        = 0.f;
    }

    indiRun.attSpNed  = (fp_quaternion_t){ 1.f, 0.f, 0.f, 0.f };
    indiRun.attErrBody = (fp_quaternion_t){ 1.f, 0.f, 0.f, 0.f };

    for (int j = 0; j < MAXV; j++)
        indiRun.dv[j] = 0.f;

    /* Timing */
    gyro.targetLooptime = CFG_LOOP_US;
    indiRun.dT = gyro.targetLooptime * 1e-6f;
    indiRun.indiFrequency = 1.0f / indiRun.dT;
    indiRun.attExecCounter = 0;
    indiRun.nanCounter = 0;

    indiRun.bypassControl    = false;
    indiRun.controlAttitude  = false;  /* all tests use rate-only */
    indiRun.trackAttitudeYaw = false;

    /* Filters */
    for (int axis = FD_ROLL; axis <= FD_YAW; axis++) {
        biquadFilterInitLPF(&indiRun.rateFilter[axis], indiRun.imuSyncLp2Hz, gyro.targetLooptime);
        biquadFilterInitLPF(&indiRun.spfFilter[axis],  indiRun.imuSyncLp2Hz, gyro.targetLooptime);
    }
    for (int i = 0; i < indiRun.actNum; i++) {
        pt1FilterInit(&indiRun.uLagFilter[i],
                      pt1FilterGain(1.f / (2.f * M_PIf * indiRun.actTimeConstS[i]), indiRun.dT));
        biquadFilterInitLPF(&indiRun.uStateFilter[i], indiRun.imuSyncLp2Hz, gyro.targetLooptime);
        biquadFilterInitLPF(&indiRun.omegaFilter[i],  indiRun.imuSyncLp2Hz, gyro.targetLooptime);
    }

    /* Sensor scaling — set acc_1G_rec so that accADCafterRpm is in g-units */
    acc.dev.acc_1G_rec = 1.0f;   /* we supply accel in g directly */

    /* gyro feedback (for rate error in getAlphaSpBody) */
    memset(gyro.gyroADCf, 0, sizeof(gyro.gyroADCf));
    memset(gyro.gyroADCafterRpm, 0, sizeof(gyro.gyroADCafterRpm));

    /* Motor state statics */
    reset_getMotorCommands_state();
}

/* ------------------------------------------------------------------ */
/*  Per-step helper: set sensor inputs                                */
/* ------------------------------------------------------------------ */

static void set_sensors(const float gyro_dps[3],   /* deg/s */
                        const float accel_g[3],     /* g-units */
                        const float rateSp_rads[3], /* rad/s */
                        float spfSp_z)              /* N/kg */
{
    /* gyro.gyroADCafterRpm is in deg/s in the real firmware */
    gyro.gyroADCafterRpm[0] = gyro_dps[0];
    gyro.gyroADCafterRpm[1] = gyro_dps[1];
    gyro.gyroADCafterRpm[2] = gyro_dps[2];

    /* gyroADCf is the main-loop filtered gyro; set equal for tests */
    gyro.gyroADCf[0] = gyro_dps[0];
    gyro.gyroADCf[1] = gyro_dps[1];
    gyro.gyroADCf[2] = gyro_dps[2];

    /* accel after RPM filter — we supply in g so acc_1G_rec = 1 */
    acc_accADCafterRpm[0] = accel_g[0];
    acc_accADCafterRpm[1] = accel_g[1];
    acc_accADCafterRpm[2] = accel_g[2];

    /* Rate setpoint — set directly (bypass getSetpoints) */
    indiRun.rateSpBody.V.X = rateSp_rads[0];
    indiRun.rateSpBody.V.Y = rateSp_rads[1];
    indiRun.rateSpBody.V.Z = rateSp_rads[2];

    /* Specific force setpoint z */
    indiRun.spfSpBody.V.X = 0.f;
    indiRun.spfSpBody.V.Y = 0.f;
    indiRun.spfSpBody.V.Z = spfSp_z;
}

/* ------------------------------------------------------------------ */
/*  CSV output                                                         */
/* ------------------------------------------------------------------ */

static void print_csv_header(void) {
    printf("test_case,step,"
           "rateDot_fs_x,rateDot_fs_y,rateDot_fs_z,"
           "spf_fs_x,spf_fs_y,spf_fs_z,"
           "omegaDot_fs_0,omegaDot_fs_1,omegaDot_fs_2,omegaDot_fs_3,"
           "uState_fs_0,uState_fs_1,uState_fs_2,uState_fs_3,"
           "dv_0,dv_1,dv_2,dv_3,dv_4,dv_5,"
           "u_0,u_1,u_2,u_3,"
           "d_0,d_1,d_2,d_3\n");
}

static void print_csv_row(const char* test_name, int step) {
    printf("%s,%d,", test_name, step);
    /* rateDot_fs */
    printf("%.9e,%.9e,%.9e,",
           indiRun.rateDot_fs.A[0], indiRun.rateDot_fs.A[1], indiRun.rateDot_fs.A[2]);
    /* spf_fs */
    printf("%.9e,%.9e,%.9e,",
           indiRun.spf_fs.A[0], indiRun.spf_fs.A[1], indiRun.spf_fs.A[2]);
    /* omegaDot_fs */
    printf("%.9e,%.9e,%.9e,%.9e,",
           indiRun.omegaDot_fs[0], indiRun.omegaDot_fs[1],
           indiRun.omegaDot_fs[2], indiRun.omegaDot_fs[3]);
    /* uState_fs */
    printf("%.9e,%.9e,%.9e,%.9e,",
           indiRun.uState_fs[0], indiRun.uState_fs[1],
           indiRun.uState_fs[2], indiRun.uState_fs[3]);
    /* dv */
    printf("%.9e,%.9e,%.9e,%.9e,%.9e,%.9e,",
           indiRun.dv[0], indiRun.dv[1], indiRun.dv[2],
           indiRun.dv[3], indiRun.dv[4], indiRun.dv[5]);
    /* u */
    printf("%.9e,%.9e,%.9e,%.9e,",
           indiRun.u[0], indiRun.u[1], indiRun.u[2], indiRun.u[3]);
    /* d */
    printf("%.9e,%.9e,%.9e,%.9e\n",
           indiRun.d[0], indiRun.d[1], indiRun.d[2], indiRun.d[3]);
}

/* ------------------------------------------------------------------ */
/*  Run a single time step                                             */
/* ------------------------------------------------------------------ */

static void run_step(const char* test_name, int step,
                     const float gyro_dps[3], const float accel_g[3],
                     const float rateSp_rads[3], float spfSp_z,
                     bool doIndi_flag)
{
    /* Configure doIndi by setting arming + ground flags */
    if (doIndi_flag) {
        ENABLE_ARMING_FLAG(ARMED);
        stub_touchingGround = false;
    } else {
        armingFlags = 0;
        stub_touchingGround = true;
    }

    set_sensors(gyro_dps, accel_g, rateSp_rads, spfSp_z);

    /* Compute angular acceleration setpoint (rate PD) */
    getAlphaSpBody();

    /* Core INDI allocation + motor commands */
    getMotorCommands();

    /* Actuator state update — uses the d[] that getMotorCommands computed */
    indiUpdateActuatorState(indiRun.d);

    /* Emit row */
    print_csv_row(test_name, step);
}

/* ------------------------------------------------------------------ */
/*  Test cases                                                         */
/* ------------------------------------------------------------------ */

static void test_hover_steady(void) {
    initIndiRuntime();
    float gyro_dps[3]     = {0.f, 0.f, 0.f};
    float accel_g[3]      = {0.f, 0.f, -1.f};
    float rateSp_rads[3]  = {0.f, 0.f, 0.f};
    float spfSp_z         = -GRAVITYf;

    for (int step = 0; step < 50; step++) {
        run_step("hover_steady", step, gyro_dps, accel_g, rateSp_rads,
                 spfSp_z, true);
    }
}

static void test_roll_step(void) {
    initIndiRuntime();
    float gyro_dps[3]     = {0.f, 0.f, 0.f};
    float accel_g[3]      = {0.f, 0.f, -1.f};
    float rateSp_rads[3]  = {2.0f, 0.f, 0.f};   /* 2 rad/s roll step */
    float spfSp_z         = -GRAVITYf;

    for (int step = 0; step < 50; step++) {
        run_step("roll_step", step, gyro_dps, accel_g, rateSp_rads,
                 spfSp_z, true);
    }
}

static void test_ground_ndi(void) {
    initIndiRuntime();
    float gyro_dps[3]     = {0.f, 0.f, 0.f};
    float accel_g[3]      = {0.f, 0.f, -1.f};
    float rateSp_rads[3]  = {0.f, 0.f, 0.f};
    float spfSp_z         = -5.0f;

    for (int step = 0; step < 20; step++) {
        run_step("ground_ndi", step, gyro_dps, accel_g, rateSp_rads,
                 spfSp_z, false);   /* doIndi = false */
    }
}

static void test_saturation(void) {
    initIndiRuntime();
    float gyro_dps[3]     = {0.f, 0.f, 0.f};
    float accel_g[3]      = {0.f, 0.f, -1.f};
    /* Large roll + pitch to saturate motors */
    float rateSp_rads[3]  = {15.0f, 15.0f, 0.f};
    float spfSp_z         = -GRAVITYf;

    for (int step = 0; step < 50; step++) {
        run_step("saturation", step, gyro_dps, accel_g, rateSp_rads,
                 spfSp_z, true);
    }
}

/* 5. Combined roll + pitch + yaw command */
static void test_combined_axes(void) {
    initIndiRuntime();
    float gyro_dps[3]     = {0.f, 0.f, 0.f};
    float accel_g[3]      = {0.f, 0.f, -1.f};
    float rateSp_rads[3]  = {3.0f, -2.0f, 1.5f};
    float spfSp_z         = -GRAVITYf;

    for (int step = 0; step < 50; step++) {
        run_step("combined_axes", step, gyro_dps, accel_g, rateSp_rads,
                 spfSp_z, true);
    }
}

/* 6. Changing commands over time — ramp then reverse */
static void test_ramp_command(void) {
    initIndiRuntime();
    float accel_g[3] = {0.f, 0.f, -1.f};

    for (int step = 0; step < 80; step++) {
        float t = step / 8000.f;
        float roll_sp;
        if (step < 40) {
            roll_sp = step * 0.2f;       /* ramp up: 0 → 8 rad/s over 40 steps */
        } else {
            roll_sp = (80 - step) * 0.2f; /* ramp down: 8 → 0 rad/s */
        }
        float gyro_dps[3]    = {0.f, 0.f, 0.f};
        float rateSp_rads[3] = {roll_sp, 0.f, 0.f};
        run_step("ramp_command", step, gyro_dps, accel_g, rateSp_rads,
                 -GRAVITYf, true);
    }
}

/* 7. Nonzero gyro — vehicle spinning, tests the incremental feedback */
static void test_spinning_vehicle(void) {
    initIndiRuntime();
    /* Vehicle is spinning at ~100 deg/s in roll, ~50 in pitch */
    float gyro_dps[3]     = {100.f, 50.f, -20.f};
    float accel_g[3]      = {0.f, 0.f, -1.f};
    float rateSp_rads[3]  = {0.f, 0.f, 0.f};  /* want to stop spinning */
    float spfSp_z         = -GRAVITYf;

    for (int step = 0; step < 50; step++) {
        run_step("spinning_vehicle", step, gyro_dps, accel_g, rateSp_rads,
                 spfSp_z, true);
    }
}

/* 8. Nonzero accel — tilted vehicle (not level hover) */
static void test_tilted_accel(void) {
    initIndiRuntime();
    float gyro_dps[3]     = {0.f, 0.f, 0.f};
    /* Tilted ~30 deg: accel has lateral component */
    float accel_g[3]      = {0.0f, 0.5f, -0.866f};
    float rateSp_rads[3]  = {0.f, 0.f, 0.f};
    float spfSp_z         = -GRAVITYf;

    for (int step = 0; step < 50; step++) {
        run_step("tilted_accel", step, gyro_dps, accel_g, rateSp_rads,
                 spfSp_z, true);
    }
}

/* 9. Asymmetric motor limits — motor 0 limited to 80% */
static void test_asymmetric_limits(void) {
    initIndiRuntime();
    indiRun.actLimit[0] = 0.8f;  /* M0 limited */
    indiRun.actLimit[1] = 1.0f;
    indiRun.actLimit[2] = 1.0f;
    indiRun.actLimit[3] = 1.0f;

    float gyro_dps[3]     = {0.f, 0.f, 0.f};
    float accel_g[3]      = {0.f, 0.f, -1.f};
    float rateSp_rads[3]  = {5.0f, 3.0f, 0.f};
    float spfSp_z         = -GRAVITYf;

    for (int step = 0; step < 50; step++) {
        run_step("asymmetric_limits", step, gyro_dps, accel_g, rateSp_rads,
                 spfSp_z, true);
    }
}

/* 10. Near-hover small corrections — typical flight ops */
static void test_small_corrections(void) {
    initIndiRuntime();
    /* Small gyro perturbation, small rate command */
    float gyro_dps[3]     = {2.f, -1.f, 0.5f};
    float accel_g[3]      = {0.01f, -0.02f, -0.998f};
    float rateSp_rads[3]  = {0.05f, -0.03f, 0.01f};
    float spfSp_z         = -GRAVITYf;

    for (int step = 0; step < 50; step++) {
        run_step("small_corrections", step, gyro_dps, accel_g, rateSp_rads,
                 spfSp_z, true);
    }
}

/* 11. Rapid setpoint reversal — tests warmstart switching */
static void test_setpoint_reversal(void) {
    initIndiRuntime();
    float accel_g[3] = {0.f, 0.f, -1.f};

    for (int step = 0; step < 60; step++) {
        float roll_sp;
        if (step < 20) {
            roll_sp = 5.0f;    /* positive roll */
        } else if (step < 40) {
            roll_sp = -5.0f;   /* sudden reversal */
        } else {
            roll_sp = 0.0f;    /* back to zero */
        }
        float gyro_dps[3]    = {0.f, 0.f, 0.f};
        float rateSp_rads[3] = {roll_sp, 0.f, 0.f};
        run_step("setpoint_reversal", step, gyro_dps, accel_g, rateSp_rads,
                 -GRAVITYf, true);
    }
}

/* 12. doIndi transition — ground to air */
static void test_doindi_transition(void) {
    initIndiRuntime();
    float gyro_dps[3]     = {0.f, 0.f, 0.f};
    float accel_g[3]      = {0.f, 0.f, -1.f};
    float rateSp_rads[3]  = {0.f, 0.f, 0.f};

    for (int step = 0; step < 60; step++) {
        /* First 20 steps: on ground (NDI), ramp up thrust */
        /* Steps 20-60: in air (INDI), hover thrust */
        bool doIndi = (step >= 20);
        float spfSp_z;
        if (step < 20) {
            spfSp_z = -5.0f + step * (-4.81f / 20.f); /* ramp from -5 to -9.81 */
        } else {
            spfSp_z = -GRAVITYf;
        }
        run_step("doindi_transition", step, gyro_dps, accel_g, rateSp_rads,
                 spfSp_z, doIndi);
    }
}

/* 13. Changing gyro — ramps over time, produces nonzero rateDot every frame.
 *     This is the core INDI finite-difference test that no constant-gyro test covers. */
static void test_changing_gyro(void) {
    initIndiRuntime();
    float accel_g[3]      = {0.f, 0.f, -1.f};
    float rateSp_rads[3]  = {0.f, 0.f, 0.f};

    for (int step = 0; step < 80; step++) {
        /* Gyro ramps: 0→200 deg/s in roll over 40 steps, then holds.
         * Also adds a pitch oscillation to test multi-axis derivative. */
        float roll_dps;
        if (step < 40) {
            roll_dps = step * 5.0f;      /* 0 → 200 deg/s */
        } else {
            roll_dps = 200.0f;           /* hold at 200 deg/s */
        }
        float pitch_dps = 30.0f * sinf(step * 0.3f); /* oscillating pitch */
        float gyro_dps[3] = {roll_dps, pitch_dps, 0.f};

        run_step("changing_gyro", step, gyro_dps, accel_g, rateSp_rads,
                 -GRAVITYf, true);
    }
}

/* 14. G2 active — nonzero G2 yaw with hover RPM fallback.
 *     Tests the combined G1+G2 matrix and omega_inv path. */
static void test_g2_active(void) {
    initIndiRuntime();

    /* Set nonzero G2 yaw values — same sign as G1 yaw (NED convention).
     * G2 yaw for CW motors = negative, CCW = positive in NED.
     * Typical values: ~1e-4 to 1e-3 in acceleration space.
     * Stored in indiRun.actG2 directly (already in runtime units). */
    indiRun.actG2[0][0] = 0.f; /* roll G2 = 0 */
    indiRun.actG2[1][0] = 0.f; /* pitch G2 = 0 */
    indiRun.actG2[2][0] = -0.001f; /* yaw G2: M0 CW → negative in NED */
    indiRun.actG2[2][1] =  0.001f; /* M1 CCW → positive */
    indiRun.actG2[2][2] =  0.001f; /* M2 CCW → positive */
    indiRun.actG2[2][3] = -0.001f; /* M3 CW → negative */

    /* Motor RPM uses hover fallback (no dshot telemetry in test) */
    /* omega_fs is set to actHoverOmega in indiUpdateActuatorState fallback */
    for (int i = 0; i < 4; i++) {
        indiRun.omega[i]    = indiRun.actHoverOmega[i];
        indiRun.omega_fs[i] = indiRun.actHoverOmega[i];
    }

    float gyro_dps[3]     = {0.f, 0.f, 0.f};
    float accel_g[3]      = {0.f, 0.f, -1.f};
    float rateSp_rads[3]  = {0.f, 0.f, 2.0f}; /* yaw command to exercise G2 path */
    float spfSp_z         = -GRAVITYf;

    for (int step = 0; step < 50; step++) {
        run_step("g2_active", step, gyro_dps, accel_g, rateSp_rads,
                 spfSp_z, true);
    }
}

/* ------------------------------------------------------------------ */
/*  main                                                               */
/* ------------------------------------------------------------------ */

int main(void) {
    print_csv_header();
    test_hover_steady();
    test_roll_step();
    test_ground_ndi();
    test_saturation();
    test_combined_axes();
    test_ramp_command();
    test_spinning_vehicle();
    test_tilted_accel();
    test_asymmetric_limits();
    test_small_corrections();
    test_setpoint_reversal();
    test_doindi_transition();
    test_changing_gyro();
    test_g2_active();
    return 0;
}
