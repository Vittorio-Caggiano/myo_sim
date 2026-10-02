// MuJoCo passive plugin "myo.muscles": De Groote-Fregly 2016 muscles (compliant/rigid tendon, or MuJoCo-native
// curves) on polynomial (function-based) or native tendon paths, with Umberger metabolics, plus Hunt-Crossley
// contacts. Forces are added to qfrc_passive; muscle states (activation, normalized fiber length) live in
// plugin_state and are advanced with explicit Euler in the plugin `advance` callback.
//
// Muscle and contact models follow Bolt (https://github.com/willwng/bolt, William Wang et al., Apache-2.0);
// see NOTICE.
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <mujoco/mjdata.h>
#include <mujoco/mjmodel.h>
#include <mujoco/mjplugin.h>
#include <mujoco/mujoco.h>

#include "myo_muscle.h"
#include "myo_contact.h"

#define MYO_NSTATE_PER_MUSCLE 10
#define MYO_NSTATE_CONTACT 5   // grf xyz, nacon, ncollision (appended after the muscle block when contacts exist)
#define MYO_NACONMAX_C 1024
void myo_geoms_from_flat(int ngeom, const double* f, MyoGeom* g);
enum { S_ACT = 0, S_LN, S_FORCE, S_LEN, S_VEL, S_ACT_DOT, S_LN_DOT, S_METAB, S_FIBER_VEL, S_NORM_LEN };

typedef struct {
  double p[MYO_NPARAM];
  int family;    // 0: DGF (Bolt), 1: MuJoCo native curves
  int ctrl_id;
  int path_type; // 0: polynomial, 1: native tendon, 2: polynomial force + point-path (tendon) length and force
                 //    (Bolt as released: muscle 0 when every muscle has a function-based path)
  int dim, order, nterm;
  int qposadr[16], dofadr[16];
  int* exps;
  double* coef;
  int tendon_id;
  double gainprm[9], lengthrange[2], acc0, dynprm[3];
} MyoMuscle;

typedef struct {
  int qposadr, dofadr;
  double q_low, q_up, k_low, k_up, damping, transition;
} MyoLimit;  // Bolt / OpenSim CoordinateLimitForce (SI units: rad, N m/rad, N m s/rad)

typedef struct {
  int n;
  int met_flags;
  double aerobic, effort;
  MyoMuscle* mus;
  int nlimit;
  MyoLimit* lim;
  int ngeom, npair, naconmax;
  MyoGeom* geom;
  int* pairs;
  MyoContact* con;
} MyoSystem;

static double step_function(double x, double t0, double t1, double v0, double v1) {
  if (x <= t0) return v0;
  if (x >= t1) return v1;
  double t = (x - t0) / (t1 - t0);
  return v0 + t * t * (3.0 - 2.0 * t) * (v1 - v0);
}

static MyoSystem* load_spec(const char* path) {
  FILE* f = fopen(path, "r");
  if (!f) { mju_warning("myo.muscles: cannot open spec '%s'", path); return NULL; }
  MyoSystem* s = (MyoSystem*)calloc(1, sizeof(MyoSystem));
  if (fscanf(f, "%d %d %lf %lf", &s->n, &s->met_flags, &s->aerobic, &s->effort) != 4) goto fail;
  s->mus = (MyoMuscle*)calloc(s->n, sizeof(MyoMuscle));
  for (int i = 0; i < s->n; i++) {
    MyoMuscle* m = s->mus + i;
    for (int k = 0; k < MYO_NPARAM; k++) if (fscanf(f, "%lf", &m->p[k]) != 1) goto fail;
    if (fscanf(f, "%d %d %d", &m->family, &m->ctrl_id, &m->path_type) != 3) goto fail;
    if (m->path_type == 0 || m->path_type == 2) {
      if (fscanf(f, "%d %d", &m->dim, &m->order) != 2) goto fail;
      for (int j = 0; j < m->dim; j++) if (fscanf(f, "%d", &m->qposadr[j]) != 1) goto fail;
      for (int j = 0; j < m->dim; j++) if (fscanf(f, "%d", &m->dofadr[j]) != 1) goto fail;
      m->nterm = myo_poly_nterm(m->dim, m->order);
      m->exps = (int*)malloc(sizeof(int) * m->nterm * m->dim);
      myo_poly_exponents(m->dim, m->order, m->exps);
      m->coef = (double*)malloc(sizeof(double) * m->nterm);
      for (int k = 0; k < m->nterm; k++) if (fscanf(f, "%lf", &m->coef[k]) != 1) goto fail;
    } else {
      if (fscanf(f, "%d", &m->tendon_id) != 1) goto fail;
    }
    if (m->path_type == 2 && fscanf(f, "%d", &m->tendon_id) != 1) goto fail;
    if (m->family == 1) {
      for (int k = 0; k < 9; k++) if (fscanf(f, "%lf", &m->gainprm[k]) != 1) goto fail;
      if (fscanf(f, "%lf %lf %lf", &m->lengthrange[0], &m->lengthrange[1], &m->acc0) != 3) goto fail;
      for (int k = 0; k < 3; k++) if (fscanf(f, "%lf", &m->dynprm[k]) != 1) goto fail;
    }
  }
  if (fscanf(f, "%d", &s->nlimit) != 1) s->nlimit = 0;
  if (s->nlimit > 0) {
    s->lim = (MyoLimit*)calloc(s->nlimit, sizeof(MyoLimit));
    for (int i = 0; i < s->nlimit; i++) {
      MyoLimit* l = s->lim + i;
      if (fscanf(f, "%d %d %lf %lf %lf %lf %lf %lf", &l->qposadr, &l->dofadr, &l->q_low, &l->q_up, &l->k_low, &l->k_up,
                 &l->damping, &l->transition) != 8) goto fail;
    }
  }
  if (fscanf(f, "%d %d %d", &s->ngeom, &s->npair, &s->naconmax) != 3) { s->ngeom = 0; s->npair = 0; }
  if (s->ngeom > 0) {
    double* gf = (double*)malloc(sizeof(double) * 26 * s->ngeom);
    for (int i = 0; i < 26 * s->ngeom; i++) if (fscanf(f, "%lf", gf + i) != 1) { free(gf); goto fail; }
    s->geom = (MyoGeom*)calloc(s->ngeom, sizeof(MyoGeom));
    myo_geoms_from_flat(s->ngeom, gf, s->geom);
    free(gf);
    s->pairs = (int*)malloc(sizeof(int) * 2 * s->npair);
    for (int i = 0; i < 2 * s->npair; i++) if (fscanf(f, "%d", s->pairs + i) != 1) goto fail;
    if (s->naconmax > MYO_NACONMAX_C) s->naconmax = MYO_NACONMAX_C;
    s->con = (MyoContact*)calloc(s->naconmax, sizeof(MyoContact));
  }
  fclose(f);
  return s;
fail:
  fclose(f);
  mju_warning("myo.muscles: malformed spec '%s'", path);
  return NULL;
}

static void free_spec(MyoSystem* s) {
  if (!s) return;
  for (int i = 0; i < s->n; i++) { free(s->mus[i].exps); free(s->mus[i].coef); }
  free(s->mus);
  free(s->lim);
  free(s->geom); free(s->pairs); free(s->con);
  free(s);
}

static int myo_nstate(const mjModel* m, int instance) {
  MyoSystem* s = load_spec(mj_getPluginConfig(m, instance, "spec"));
  int n = s ? s->n : 0, nc = (s && s->ngeom > 0) ? MYO_NSTATE_CONTACT : 0;
  free_spec(s);
  return MYO_NSTATE_PER_MUSCLE * n + nc;
}

static int myo_init(const mjModel* m, mjData* d, int instance) {
  MyoSystem* s = load_spec(mj_getPluginConfig(m, instance, "spec"));
  if (!s) return -1;
  d->plugin_data[instance] = (uintptr_t)s;
  return 0;
}

static void myo_destroy(mjData* d, int instance) {
  free_spec((MyoSystem*)d->plugin_data[instance]);
  d->plugin_data[instance] = 0;
}

static void myo_reset(const mjModel* m, mjtNum* state, void* data, int instance) {
  MyoSystem* s = (MyoSystem*)data;
  if (!s) return;
  for (int i = 0; i < s->n; i++) {
    mjtNum* st = state + MYO_NSTATE_PER_MUSCLE * i;
    memset(st, 0, sizeof(mjtNum) * MYO_NSTATE_PER_MUSCLE);
    st[S_ACT] = s->mus[i].family == 0 ? s->mus[i].p[MYO_P_AMIN] : 0.0;
    st[S_LN] = 1.0;
  }
}

static void myo_limits(const mjModel* m, mjData* d, MyoSystem* s);
static void myo_contacts(const mjModel* m, mjData* d, MyoSystem* s, mjtNum* cstate);

static void myo_compute(const mjModel* m, mjData* d, int instance, int capability_bit) {
  MyoSystem* s = (MyoSystem*)d->plugin_data[instance];
  mjtNum* state = d->plugin_state + m->plugin_stateadr[instance];
  for (int i = 0; i < s->n; i++) {
    MyoMuscle* mu = s->mus + i;
    mjtNum* st = state + MYO_NSTATE_PER_MUSCLE * i;
    double L, V, grad[16];
    // ---- path
    if (mu->path_type == 0 || mu->path_type == 2) {
      double q[16];
      for (int j = 0; j < mu->dim; j++) q[j] = d->qpos[mu->qposadr[j]];
      L = myo_poly_eval(mu->dim, mu->order, mu->nterm, mu->exps, mu->coef, q, grad);
      V = 0.0;
      for (int j = 0; j < mu->dim; j++) V += grad[j] * d->qvel[mu->dofadr[j]];
      if (mu->path_type == 2) { L = d->ten_length[mu->tendon_id]; V = d->ten_velocity[mu->tendon_id]; }
    } else {
      L = d->ten_length[mu->tendon_id];
      V = d->ten_velocity[mu->tendon_id];
    }
    // ---- muscle
    double u = mju_clip(d->ctrl[mu->ctrl_id], 0.0, 1.0);
    double act = st[S_ACT];
    double F;
    if (mu->family == 0) {
      double out[MYO_NOUT];
      myo_contraction(L, V, act, st[S_LN], mu->p, out);
      F = out[MYO_O_TENDON_FORCE];
      st[S_ACT_DOT] = myo_act_dot(act, u, mu->p);
      st[S_LN_DOT] = out[MYO_O_LN_DOT];
      st[S_METAB] = (s->met_flags & MYO_MET_OFF) ? 0.0
                    : myo_metabolics(act, u, out, mu->p, s->met_flags, s->aerobic, s->effort);
      st[S_FIBER_VEL] = out[MYO_O_FIBER_VEL];
      st[S_NORM_LEN] = out[MYO_O_NORM_FIBER_LENGTH];
    } else {
      double gain = mju_muscleGain(L, V, mu->lengthrange, mu->acc0, mu->gainprm);
      double bias = mju_muscleBias(L, mu->lengthrange, mu->acc0, mu->gainprm);
      F = -(gain * act + bias);
      st[S_ACT_DOT] = mju_muscleDynamics(u, act, mu->dynprm);
      st[S_LN_DOT] = 0.0; st[S_METAB] = 0.0; st[S_FIBER_VEL] = 0.0; st[S_NORM_LEN] = 0.0;
    }
    st[S_FORCE] = F; st[S_LEN] = L; st[S_VEL] = V;
    // ---- generalized force: qfrc = -F * dL/dq
    if (mu->path_type == 0 || mu->path_type == 2) {
      for (int j = 0; j < mu->dim; j++) d->qfrc_passive[mu->dofadr[j]] += -F * grad[j];
    }
    if (mu->path_type != 0) {
      int adr = m->ten_J_rowadr[mu->tendon_id], nnz = m->ten_J_rownnz[mu->tendon_id];
      for (int k = 0; k < nnz; k++) d->qfrc_passive[m->ten_J_colind[adr + k]] += -F * d->ten_J[adr + k];
    }
  }
  myo_limits(m, d, s);
  if (s->ngeom > 0) myo_contacts(m, d, s, state + MYO_NSTATE_PER_MUSCLE * s->n);
}

static void myo_limits(const mjModel* m, mjData* d, MyoSystem* s) {
  for (int i = 0; i < s->nlimit; i++) {
    MyoLimit* l = s->lim + i;
    double q = d->qpos[l->qposadr], qd = d->qvel[l->dofadr];
    double w_up = step_function(q, l->q_up, l->q_up + l->transition, 0.0, 1.0);
    double w_low = step_function(q, l->q_low - l->transition, l->q_low, 1.0, 0.0);
    double f = -(w_up * l->k_up) * (q - l->q_up) + (w_low * l->k_low) * (l->q_low - q) - l->damping * (w_up + w_low) * qd;
    d->qfrc_passive[l->dofadr] += f;
  }
}

// Bolt Hunt-Crossley contacts; forces applied as Cartesian point forces through the body Jacobians
static void myo_contacts(const mjModel* m, mjData* d, MyoSystem* s, mjtNum* cstate) {
  double gpos_buf[3 * 64], grot_buf[9 * 64];
  int heap = s->ngeom > 64;
  double* gpos = heap ? (double*)malloc(sizeof(double) * 3 * s->ngeom) : gpos_buf;
  double* grot = heap ? (double*)malloc(sizeof(double) * 9 * s->ngeom) : grot_buf;
  myo_geom_poses(s->ngeom, s->geom, d->xpos, d->xquat, gpos, grot);
  int ncol = 0;
  int nacon = myo_collide(s->ngeom, s->geom, gpos, grot, s->npair, s->pairs, s->naconmax, &ncol, s->con);
  double grf[3] = {0, 0, 0};
  for (int i = 0; i < nacon; i++) {
    MyoContact* c = s->con + i;
    int b1 = s->geom[c->g1].body, b2 = s->geom[c->g2].body;
    double v[2][3];
    int bb[2] = {b1, b2};
    for (int k = 0; k < 2; k++) {
      double res[6];
      mju_transformSpatial(res, d->cvel + 6 * bb[k], 0, c->pos, d->subtree_com + 3 * m->body_rootid[bb[k]], 0);
      v[k][0] = res[3]; v[k][1] = res[4]; v[k][2] = res[5];
    }
    double force[3], mforce[3], zero[3] = {0, 0, 0};
    if (!myo_hc_force(c, v[0], v[1], force)) continue;
    mju_scl3(mforce, force, -1.0);
    if (b1 > 0) mj_applyFT(m, d, mforce, zero, c->pos, b1, d->qfrc_passive);
    if (b2 > 0) mj_applyFT(m, d, force, zero, c->pos, b2, d->qfrc_passive);
    if (b1 == 0) mju_addTo3(grf, force);
    else if (b2 == 0) mju_subFrom3(grf, force);
  }
  cstate[0] = grf[0]; cstate[1] = grf[1]; cstate[2] = grf[2]; cstate[3] = nacon; cstate[4] = ncol;
  if (heap) { free(gpos); free(grot); }
}

static void myo_advance(const mjModel* m, mjData* d, int instance) {
  MyoSystem* s = (MyoSystem*)d->plugin_data[instance];
  mjtNum* state = d->plugin_state + m->plugin_stateadr[instance];
  double h = m->opt.timestep;
  for (int i = 0; i < s->n; i++) {
    MyoMuscle* mu = s->mus + i;
    mjtNum* st = state + MYO_NSTATE_PER_MUSCLE * i;
    if (mu->family == 0) {
      st[S_ACT] = mju_clip(st[S_ACT] + h * st[S_ACT_DOT], mu->p[MYO_P_AMIN], mu->p[MYO_P_AMAX]);
      st[S_LN] = (mu->p[MYO_P_RIGID] > 0.5) ? 0.0
                 : mju_clip(st[S_LN] + h * st[S_LN_DOT], mu->p[MYO_P_LMIN], mu->p[MYO_P_LMAX]);
    } else {
      st[S_ACT] = mju_clip(st[S_ACT] + h * st[S_ACT_DOT], 0.0, 1.0);
    }
  }
}

mjPLUGIN_LIB_INIT(myo_muscles) {
  mjpPlugin plugin;
  mjp_defaultPlugin(&plugin);
  plugin.name = "myo.muscles";
  plugin.capabilityflags |= mjPLUGIN_PASSIVE;
  static const char* attributes[] = {"spec"};
  plugin.nattribute = 1;
  plugin.attributes = attributes;
  plugin.nstate = myo_nstate;
  plugin.init = myo_init;
  plugin.destroy = myo_destroy;
  plugin.reset = myo_reset;
  plugin.compute = myo_compute;
  plugin.advance = myo_advance;
  mjp_registerPlugin(&plugin);
}
