// MyoSim x Bolt V&V: C implementation of the spec (ref/spec.py). float64.
#pragma once
#ifdef __cplusplus
extern "C" {
#endif

// parameter layout (must match ref/spec.py PARAM_NAMES)
enum {
  MYO_P_F0 = 0, MYO_P_LOPT, MYO_P_LTS, MYO_P_ALPHA0, MYO_P_VMAX, MYO_P_DAMPING, MYO_P_E0, MYO_P_TSTRAIN,
  MYO_P_WSCALE, MYO_P_LMIN, MYO_P_LMAX, MYO_P_TAU_ACT, MYO_P_TAU_DEACT, MYO_P_ACT_SMOOTH, MYO_P_AMIN,
  MYO_P_AMAX, MYO_P_SPEC_TENSION, MYO_P_DENSITY, MYO_P_SLOW_TWITCH, MYO_P_RIGID, MYO_P_ACT_MODEL,
  MYO_NPARAM
};

// contraction output layout
enum {
  MYO_O_TENDON_FORCE = 0, MYO_O_NORM_FIBER_LENGTH, MYO_O_NORM_FIBER_VEL, MYO_O_FIBER_VEL, MYO_O_ACTIVE_FORCE,
  MYO_O_FAL, MYO_O_FV, MYO_O_FPE, MYO_O_PENNATION, MYO_O_LN_DOT, MYO_O_TENDON_LENGTH, MYO_O_CLAMPED,
  MYO_NOUT
};

// metabolic flags bitfield
enum {
  MYO_MET_BHARGAVA = 1, MYO_MET_NEG_WORK = 2, MYO_MET_FORBID_NEG = 4, MYO_MET_MIN_HEAT = 8,
  MYO_MET_BOLT_VARIANT = 16,
  MYO_MET_OFF = 256  // plugin: skip metabolics (state slot stays 0); dynamics are unaffected
};

double myo_fl_active(double ln, double width_scale);
double myo_fv(double v);
double myo_fv_deriv(double v);
double myo_fv_inv(double f);
double myo_fp(double ln, double lmin, double e0);
double myo_ft(double ltn, double strain, int clamped);
double myo_ft_inv(double f, double strain);
double myo_act_dot(double a, double u, const double* p);
void myo_contraction(double path_len, double path_vel, double a, double ln_state, const double* p, double* out);
double myo_equilibrium(double path_len, double path_vel, double a, const double* p);
double myo_metabolics(double a, double u, const double* out, const double* p, int flags, double aerobic,
                      double effort);
// polynomial path: exps is (nterm x dim) int table, returns length and writes gradient (dim)
double myo_poly_eval(int dim, int order, int nterm, const int* exps, const double* coef, const double* q,
                     double* grad);
int myo_poly_nterm(int dim, int order);
void myo_poly_exponents(int dim, int order, int* exps);

// batch entry points for ctypes
void myo_curves_batch(int n, const double* x, const double* ws, const double* lmin, const double* e0,
                      const double* strain, double* out);
void myo_act_dot_batch(int n, const double* a, const double* u, const double* p, double* out);
void myo_contraction_batch(int n, const double* L, const double* V, const double* a, const double* ln,
                           const double* p, double* out);
void myo_equilibrium_batch(int n, const double* L, const double* V, const double* a, const double* p,
                           double* out);
void myo_metabolics_batch(int n, const double* a, const double* u, const double* cout, const double* p,
                          int flags, double aerobic, double effort, double* out);
void myo_poly_batch(int n, int dim, int order, const double* coef, const double* q, double* L, double* grad);

#ifdef __cplusplus
}
#endif
