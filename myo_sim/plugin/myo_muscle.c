// De Groote-Fregly 2016 muscle contraction dynamics + Umberger metabolics, float64 C implementation.
//
// Formulation follows Bolt (https://github.com/willwng/bolt, William Wang et al., Apache-2.0): clamps,
// branch order, rigid-tendon handling, damped Newton and reset equilibrium. Curve constants in
// myo_consts.h follow OpenSim's Apache-2.0 DeGrooteFregly2016Muscle. See NOTICE.
#include "myo_muscle.h"

#include <math.h>
#include <string.h>

#include "myo_consts.h"

static inline double dmin(double a, double b) { return a < b ? a : b; }
static inline double dmax(double a, double b) { return a > b ? a : b; }
static inline double dclip(double x, double lo, double hi) { return x < lo ? lo : (x > hi ? hi : x); }

static inline double gauss(double x, double b1, double b2, double b3, double b4) {
  double den = b3 + b4 * x;
  return b1 * exp(-0.5 * (x - b2) * (x - b2) / (den * den));
}

double myo_fl_active(double ln, double ws) {
  double x = (ln - 1.0) / ws + 1.0;
  return gauss(x, MYO_B11, MYO_B21, MYO_B31, MYO_B41) + gauss(x, MYO_B12, MYO_B22, MYO_B32, MYO_B42) +
         gauss(x, MYO_B13, MYO_B23, MYO_B33, MYO_B43);
}

double myo_fv(double v) {
  double t = MYO_D2 * v + MYO_D3;
  double s = sqrt(t * t + 1.0);
  double arg = (t >= 0.0) ? (t + s) : (1.0 / (s - t));
  return MYO_D1 * log(arg) + MYO_D4;
}

double myo_fv_deriv(double v) {
  double t = MYO_D2 * v + MYO_D3;
  return (MYO_D1 * MYO_D2) / sqrt(t * t + 1.0);
}

double myo_fv_inv(double f) { return (sinh((f - MYO_D4) / MYO_D1) - MYO_D3) / MYO_D2; }

double myo_fp(double ln, double lmin, double e0) {
  double offset = exp(MYO_KPE * (lmin - 1.0) / e0);
  return (exp(MYO_KPE * (ln - 1.0) / e0) - offset) / (exp(MYO_KPE) - offset);
}

static inline double tendon_kT(double strain) { return log((1.0 + MYO_C3) / MYO_C1) / (1.0 + strain - MYO_C2); }

double myo_ft(double ltn, double strain, int clamped) {
  double f = MYO_C1 * exp(tendon_kT(strain) * (ltn - MYO_C2)) - MYO_C3;
  return clamped ? dclip(f, MYO_MIN_NORM_TENDON_FORCE, MYO_MAX_NORM_TENDON_FORCE) : f;
}

double myo_ft_inv(double f, double strain) { return log((f + MYO_C3) / MYO_C1) / tendon_kT(strain) + MYO_C2; }

double myo_act_dot(double a, double u, const double* p) {
  double ta = p[MYO_P_TAU_ACT], td = p[MYO_P_TAU_DEACT], b = p[MYO_P_ACT_SMOOTH];
  if (p[MYO_P_ACT_MODEL] == 0.0) {  // DGF
    double tcf = 0.5 + 1.5 * a;
    double f = 0.5 * tanh(b * (u - a));
    double tc = (1.0 / (ta * tcf)) * (f + 0.5) + (tcf / td) * (-f + 0.5);
    return tc * (u - a);
  }
  double tau = (u > a) ? ta * (0.5 + 1.5 * a) : td / (0.5 + 1.5 * a);
  return (u - a) / tau;
}

static double pennation_angle(double ln, const double* p) {
  double lopt = p[MYO_P_LOPT], alpha0 = p[MYO_P_ALPHA0], lmin = p[MYO_P_LMIN];
  if (!(alpha0 > MYO_PENNATION_EPS)) return 0.0;
  if (!(ln > lmin)) return MYO_MAX_PENNATION_ANGLE;
  double h = lopt * sin(alpha0);
  double sin_phi = h / (ln * lopt);
  return (sin_phi < sin(MYO_MAX_PENNATION_ANGLE)) ? asin(dmin(sin_phi, 1.0)) : MYO_MAX_PENNATION_ANGLE;
}

static double rigid_fiber_length(double path_len, const double* p) {
  double lopt = p[MYO_P_LOPT], alpha0 = p[MYO_P_ALPHA0], lts = p[MYO_P_LTS];
  double at = path_len - lts;
  double min_at = p[MYO_P_LMIN] * lopt * cos(MYO_MAX_PENNATION_ANGLE);
  double h = lopt * sin(alpha0);
  double lm = (at >= min_at) ? sqrt(h * h + at * at) : min_at;
  return dclip(lm, p[MYO_P_LMIN] * lopt, p[MYO_P_LMAX] * lopt);
}

static double damped_fiber_velocity(double F0, double a, double fal, double fpe, double fse, double beta,
                                    double cphi, double* fv_out) {
  double fv0 = (fse / dmax(cphi, 0.01) - fpe) / (dmax(a, 0.01) * dmax(fal, 0.01));
  double v = dclip(myo_fv_inv(fv0), -1.0, 1.0);
  double tol = dmax(MYO_DAMPED_TOL_REL * F0, MYO_DAMPED_TOL_ABS);
  for (int i = 0; i < MYO_DAMPED_NEWTON_ITERS; i++) {
    double fvv = myo_fv(v), dfv = myo_fv_deriv(v);
    double err = F0 * (a * fal * fvv + fpe + beta * v) * cphi - fse * F0;
    double derr = F0 * (a * fal * dfv + beta) * cphi;
    // Once an iterate is not updated, every later iteration repeats it exactly: stopping is bit-identical.
    if (fabs(err) > tol && fabs(derr) > MYO_SIG_REAL) v = v - err / derr;
    else break;
  }
  *fv_out = myo_fv(v);
  return v;
}

void myo_contraction(double path_len, double path_vel, double a, double ln_state, const double* p,
                     double* out) {
  double F0 = p[MYO_P_F0], lopt = p[MYO_P_LOPT], lts = p[MYO_P_LTS], vmax = p[MYO_P_VMAX];
  double beta = p[MYO_P_DAMPING], e0 = p[MYO_P_E0], strain = p[MYO_P_TSTRAIN], ws = p[MYO_P_WSCALE];
  double lmin = p[MYO_P_LMIN];
  int rigid = p[MYO_P_RIGID] > 0.5;

  double ln = rigid ? rigid_fiber_length(path_len, p) / lopt : ln_state;
  double lm = ln * lopt;
  double phi = pennation_angle(ln, p);
  double cphi = cos(phi);
  double lt = path_len - lm * cphi;
  double ltn = lt / lts;
  double fpe = myo_fp(ln, lmin, e0);
  double fal = myo_fl_active(ln, ws);
  double fse = myo_ft(ltn, strain, 1);
  double vmax_ms = vmax * lopt;

  double vn, fvm;
  if (rigid) {
    if (lt < lts - MYO_SIG_REAL) {
      vn = 0.0; fvm = 1.0;
    } else {
      vn = (path_vel * cphi) / vmax_ms; fvm = myo_fv(vn);
    }
  } else if (beta > 0.0) {
    vn = damped_fiber_velocity(F0, a, fal, fpe, fse, beta, cphi, &fvm);
  } else {
    fvm = (fse / cphi - fpe) / (a * fal);
    vn = myo_fv_inv(fvm);
  }
  double vm = vn * vmax_ms;
  int clamped = (ln <= lmin && vn < 0.0) || (ln < lmin);
  if (clamped) { vn = 0.0; vm = 0.0; fvm = 1.0; }

  double aFm = 0.0, fmAT = 0.0;
  if (!clamped) {
    aFm = F0 * a * fal * fvm;
    double fm = aFm + F0 * fpe + F0 * beta * vn;
    if (rigid && fm < 0.0) fm = 0.0;
    fmAT = fm * cphi;
  }
  double fse_out = rigid ? fmAT / F0 : fse;

  out[MYO_O_TENDON_FORCE] = fse_out * F0;
  out[MYO_O_NORM_FIBER_LENGTH] = ln;
  out[MYO_O_NORM_FIBER_VEL] = vn;
  out[MYO_O_FIBER_VEL] = vm;
  out[MYO_O_ACTIVE_FORCE] = aFm;
  out[MYO_O_FAL] = fal;
  out[MYO_O_FV] = fvm;
  out[MYO_O_FPE] = fpe;
  out[MYO_O_PENNATION] = phi;
  out[MYO_O_LN_DOT] = rigid ? 0.0 : vm / lopt;
  out[MYO_O_TENDON_LENGTH] = lt;
  out[MYO_O_CLAMPED] = clamped;
}

static double eq_residual(double ftn, double L, double V, double a, const double* p, double* lm_out) {
  double F0 = p[MYO_P_F0], lopt = p[MYO_P_LOPT], lts = p[MYO_P_LTS];
  double ltn = myo_ft_inv(ftn, p[MYO_P_TSTRAIN]);
  double lt = ltn * lts;
  double h = lopt * sin(p[MYO_P_ALPHA0]);
  double lm_at = L - lt;
  double lm = sqrt(lm_at * lm_at + h * h);
  double ln = lm / lopt;
  double cphi = lm_at / lm;
  double phi = asin(h / lm);
  if (phi > acos(0.1)) cphi = 0.1;
  double vn = (V * cphi) / (p[MYO_P_VMAX] * lopt);
  double fiber = F0 * a * myo_fl_active(ln, p[MYO_P_WSCALE]) * myo_fv(vn) +
                 F0 * (myo_fp(ln, p[MYO_P_LMIN], p[MYO_P_E0]) + p[MYO_P_DAMPING] * vn);
  *lm_out = lm;
  return ftn - fiber * cphi / F0;
}

double myo_equilibrium(double L, double V, double a, const double* p) {
  if (p[MYO_P_RIGID] > 0.5) return 0.0;
  double lo = MYO_MIN_NORM_TENDON_FORCE, hi = MYO_MAX_NORM_TENDON_FORCE, mid = 0.5 * (lo + hi);
  double lm_lo, lm_hi, lm_mid;
  double r_lo = eq_residual(lo, L, V, a, p, &lm_lo);
  double r_hi = eq_residual(hi, L, V, a, p, &lm_hi);
  double r_mid = eq_residual(mid, L, V, a, p, &lm_mid);
  int use_lo = fabs(r_lo) < fabs(r_hi);
  double best_r = use_lo ? r_lo : r_hi, best_lm = use_lo ? lm_lo : lm_hi;
  for (int i = 0; i < MYO_EQ_BISECT_ITERS; i++) {
    if (fabs(best_r) < MYO_EQ_TOL || 0.5 * (hi - lo) < MYO_EQ_TOL) break;
    if (r_lo * r_mid > 0.0) { lo = mid; r_lo = r_mid; } else { hi = mid; }
    mid = 0.5 * (lo + hi);
    r_mid = eq_residual(mid, L, V, a, p, &lm_mid);
    if (fabs(r_mid) < fabs(best_r)) { best_r = r_mid; best_lm = lm_mid; }
  }
  return dclip(best_lm / p[MYO_P_LOPT], p[MYO_P_LMIN], p[MYO_P_LMAX]);
}

double myo_metabolics(double a, double u, const double* o, const double* p, int flags, double aerobic,
                      double effort) {
  const double PI = 3.14159265358979323846;
  double F0 = p[MYO_P_F0], lopt = p[MYO_P_LOPT], vmax = p[MYO_P_VMAX];
  double mass = (F0 / p[MYO_P_SPEC_TENSION]) * p[MYO_P_DENSITY] * lopt;
  double s = p[MYO_P_SLOW_TWITCH];
  a *= effort; u *= effort;
  double fa = dmax(effort * o[MYO_O_ACTIVE_FORCE], 0.0);
  double ln = o[MYO_O_NORM_FIBER_LENGTH], vm = o[MYO_O_FIBER_VEL], vnorm = vm / lopt, fiso = o[MYO_O_FAL];
  int bolt = flags & MYO_MET_BOLT_VARIANT;
  int neg_work = (flags & MYO_MET_NEG_WORK) != 0;
  double A = (u > a) ? u : 0.5 * (u + a);
  if (flags & MYO_MET_BHARGAVA) {
    double us = s * sin(0.5 * PI * u), uf = (1.0 - s) * (1.0 - cos(0.5 * PI * u));
    s = (u == 0.0) ? 1.0 : us / (us + uf);
  }
  double am_u = 128.0 * (1.0 - s) + 25.0;
  double AM;
  if (ln <= 1.0) AM = aerobic * pow(A, 0.6) * am_u;
  else AM = (bolt ? 1.0 : aerobic) * pow(A, 0.6) * (0.4 * am_u + 0.6 * am_u * fiso);
  double x = bolt ? ln : vnorm;
  double a_ft = 153.0 / vmax, a_st = 100.0 / (vmax / 2.5);
  double S;
  if (x <= 0.0) {
    double st = dmin(-a_st * x, 100.0);
    S = aerobic * pow(A, 2.0) * (st * s - a_ft * x * (1.0 - s));
  } else {
    S = aerobic * A * ((neg_work ? 4.0 : 0.3) * a_st * x);
  }
  if (ln > 1.0) S *= fiso;
  double W = (neg_work || vm <= 0.0) ? -fa * vm : 0.0;
  W /= mass;
  if (flags & MYO_MET_FORBID_NEG) {
    double e = AM + S + W;
    if (e < 0.0) S -= e;
  }
  double heat = AM + S;
  if ((flags & MYO_MET_MIN_HEAT) && heat < 1.0) heat = 1.0;
  return (heat + W) * mass;
}

// ---------------------------------------------------------------- polynomial paths
static void rec_exps(int dim, int order, int depth, int* cur, int* exps, int* k) {
  if (depth == dim) {
    memcpy(exps + (*k) * dim, cur, sizeof(int) * dim);
    (*k)++;
    return;
  }
  for (int i = 0; i <= order; i++) {
    cur[depth] = i;
    rec_exps(dim, order - i, depth + 1, cur, exps, k);
  }
}

int myo_poly_nterm(int dim, int order) {  // C(dim+order, order)
  long num = 1, den = 1;
  for (int i = 1; i <= order; i++) { num *= dim + i; den *= i; }
  return (int)(num / den);
}

void myo_poly_exponents(int dim, int order, int* exps) {
  int cur[16] = {0}, k = 0;
  rec_exps(dim, order, 0, cur, exps, &k);
}

double myo_poly_eval(int dim, int order, int nterm, const int* exps, const double* coef, const double* q,
                     double* grad) {
  double pw[16][16];
  for (int j = 0; j < dim; j++) {
    pw[j][0] = 1.0;
    for (int e = 1; e <= order; e++) pw[j][e] = pw[j][e - 1] * q[j];
    grad[j] = 0.0;
  }
  double L = 0.0;
  for (int k = 0; k < nterm; k++) {
    const int* e = exps + k * dim;
    double term = coef[k];
    for (int j = 0; j < dim; j++) if (e[j]) term = term * pw[j][e[j]];
    L = L + term;
    for (int j = 0; j < dim; j++) {
      if (!e[j]) continue;
      double t = coef[k] * (double)e[j];
      for (int i = 0; i < dim; i++) {
        if (i == j) { if (e[i] - 1 > 0) t = t * pw[i][e[i] - 1]; }
        else if (e[i]) t = t * pw[i][e[i]];
      }
      grad[j] = grad[j] + t;
    }
  }
  return L;
}

// ---------------------------------------------------------------- batch entry points
void myo_curves_batch(int n, const double* x, const double* ws, const double* lmin, const double* e0,
                      const double* strain, double* out) {
  for (int i = 0; i < n; i++) {
    double* o = out + 7 * i;
    o[0] = myo_fl_active(x[i], ws[i]);
    o[1] = myo_fv(x[i]);
    o[2] = myo_fv_inv(x[i]);
    o[3] = myo_fp(x[i], lmin[i], e0[i]);
    o[4] = myo_ft(x[i], strain[i], 0);
    o[5] = myo_ft_inv(x[i], strain[i]);
    o[6] = myo_fv_deriv(x[i]);
  }
}

void myo_act_dot_batch(int n, const double* a, const double* u, const double* p, double* out) {
  for (int i = 0; i < n; i++) out[i] = myo_act_dot(a[i], u[i], p + MYO_NPARAM * i);
}

void myo_contraction_batch(int n, const double* L, const double* V, const double* a, const double* ln,
                           const double* p, double* out) {
  for (int i = 0; i < n; i++) myo_contraction(L[i], V[i], a[i], ln[i], p + MYO_NPARAM * i, out + MYO_NOUT * i);
}

void myo_equilibrium_batch(int n, const double* L, const double* V, const double* a, const double* p,
                           double* out) {
  for (int i = 0; i < n; i++) out[i] = myo_equilibrium(L[i], V[i], a[i], p + MYO_NPARAM * i);
}

void myo_metabolics_batch(int n, const double* a, const double* u, const double* co, const double* p, int flags,
                          double aerobic, double effort, double* out) {
  for (int i = 0; i < n; i++)
    out[i] = myo_metabolics(a[i], u[i], co + MYO_NOUT * i, p + MYO_NPARAM * i, flags, aerobic, effort);
}

void myo_poly_batch(int n, int dim, int order, const double* coef, const double* q, double* L, double* grad) {
  int nterm = myo_poly_nterm(dim, order);
  int exps[4096 * 6];
  myo_poly_exponents(dim, order, exps);
  for (int i = 0; i < n; i++) L[i] = myo_poly_eval(dim, order, nterm, exps, coef, q + dim * i, grad + dim * i);
}
