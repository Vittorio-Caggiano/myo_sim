// Hunt-Crossley contact pipeline, float64 C implementation.
//
// Formulation follows Bolt (https://github.com/willwng/bolt, William Wang et al., Apache-2.0): broadphase
// filters, narrowphase and Hunt-Crossley force with friction. See NOTICE.
#include "myo_contact.h"

#include <math.h>
#include <string.h>

static inline double dot3(const double* a, const double* b) { return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]; }
static inline void sub3(double* r, const double* a, const double* b) { r[0] = a[0] - b[0]; r[1] = a[1] - b[1]; r[2] = a[2] - b[2]; }
static inline void add3(double* r, const double* a, const double* b) { r[0] = a[0] + b[0]; r[1] = a[1] + b[1]; r[2] = a[2] + b[2]; }
static inline void scl3(double* r, const double* a, double s) { r[0] = a[0] * s; r[1] = a[1] * s; r[2] = a[2] * s; }
static inline void cross3(double* r, const double* a, const double* b) {
  double t[3] = {a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]};
  memcpy(r, t, sizeof(t));
}
static inline double clampd(double x, double lo, double hi) { return x < lo ? lo : (x > hi ? hi : x); }
static inline void col3(double* r, const double* R, int c) { r[0] = R[c]; r[1] = R[3 + c]; r[2] = R[6 + c]; }
static inline void mulmv(double* r, const double* R, const double* v) {
  double t[3] = {R[0] * v[0] + R[1] * v[1] + R[2] * v[2], R[3] * v[0] + R[4] * v[1] + R[5] * v[2],
                 R[6] * v[0] + R[7] * v[1] + R[8] * v[2]};
  memcpy(r, t, sizeof(t));
}

static void quat2mat(const double* q, double* R) {
  double w = q[0], x = q[1], y = q[2], z = q[3];
  R[0] = 1 - 2 * (y * y + z * z); R[1] = 2 * (x * y - z * w); R[2] = 2 * (x * z + y * w);
  R[3] = 2 * (x * y + z * w); R[4] = 1 - 2 * (x * x + z * z); R[5] = 2 * (y * z - x * w);
  R[6] = 2 * (x * z - y * w); R[7] = 2 * (y * z + x * w); R[8] = 1 - 2 * (x * x + y * y);
}

static void quatmul(const double* a, const double* b, double* r) {
  double t[4] = {a[0] * b[0] - a[1] * b[1] - a[2] * b[2] - a[3] * b[3],
                 a[0] * b[1] + a[1] * b[0] + a[2] * b[3] - a[3] * b[2],
                 a[0] * b[2] - a[1] * b[3] + a[2] * b[0] + a[3] * b[1],
                 a[0] * b[3] + a[1] * b[2] - a[2] * b[1] + a[3] * b[0]};
  memcpy(r, t, sizeof(t));
}

void myo_geom_poses(int ngeom, const MyoGeom* g, const double* bpos, const double* bquat, double* gpos, double* grot) {
  for (int i = 0; i < ngeom; i++) {
    const double* bp = bpos + 3 * g[i].body;
    const double* bq = bquat + 4 * g[i].body;
    double Rb[9], q[4], t[3];
    quat2mat(bq, Rb);
    mulmv(t, Rb, g[i].pos);
    add3(gpos + 3 * i, bp, t);
    quatmul(bq, g[i].quat, q);
    double nq = sqrt(q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]);
    for (int k = 0; k < 4; k++) q[k] /= nq;
    quat2mat(q, grot + 9 * i);
  }
}

// ------------------------------------------------------------------------------------------------ broadphase
static int plane_filter(double size1, double size2, const double* xpos1, const double* xpos2, const double* xmat1,
                        const double* xmat2) {
  double d[3], n[3];
  if (size1 == 0.0) {
    sub3(d, xpos2, xpos1); col3(n, xmat1, 1);
    return dot3(d, n) <= size2;
  } else if (size2 == 0.0) {
    sub3(d, xpos1, xpos2); col3(n, xmat2, 1);
    return dot3(d, n) <= size1;
  }
  return 1;
}

static int sphere_filter(double size1, double size2, const double* xpos1, const double* xpos2) {
  double bound = size1 + size2, d[3];
  sub3(d, xpos2, xpos1);
  return dot3(d, d) <= bound * bound;
}

static int aabb_filter(const double* c1, const double* c2, const double* s1, const double* s2, const double* xpos1,
                       const double* xpos2, const double* xmat1, const double* xmat2) {
  double center1[3], center2[3];
  mulmv(center1, xmat1, c1); add3(center1, center1, xpos1);
  mulmv(center2, xmat2, c2); add3(center2, center2, xpos2);
  double max1[3] = {-1e10, -1e10, -1e10}, min1[3] = {1e10, 1e10, 1e10};
  double max2[3] = {-1e10, -1e10, -1e10}, min2[3] = {1e10, 1e10, 1e10};
  const double sign[2] = {-1.0, 1.0};
  for (int i = 0; i < 2; i++)
    for (int j = 0; j < 2; j++)
      for (int k = 0; k < 2; k++) {
        double k1[3] = {sign[i] * s1[0], sign[j] * s1[1], sign[k] * s1[2]}, p1[3];
        double k2[3] = {sign[i] * s2[0], sign[j] * s2[1], sign[k] * s2[2]}, p2[3];
        mulmv(p1, xmat1, k1); mulmv(p2, xmat2, k2);
        for (int a = 0; a < 3; a++) {
          if (p1[a] > max1[a]) max1[a] = p1[a];
          if (p1[a] < min1[a]) min1[a] = p1[a];
          if (p2[a] > max2[a]) max2[a] = p2[a];
          if (p2[a] < min2[a]) min2[a] = p2[a];
        }
      }
  for (int a = 0; a < 3; a++) {
    if (center1[a] + max1[a] < center2[a] + min2[a]) return 0;
    if (center2[a] + max2[a] < center1[a] + min1[a]) return 0;
  }
  return 1;
}

static int obb_filter(const double* c1, const double* c2, const double* s1, const double* s2, const double* xpos1,
                      const double* xpos2, const double* xmat1, const double* xmat2) {
  double xc[2][3], nrm[6][3];
  mulmv(xc[0], xmat1, c1); add3(xc[0], xc[0], xpos1);
  mulmv(xc[1], xmat2, c2); add3(xc[1], xc[1], xpos2);
  for (int k = 0; k < 3; k++) { col3(nrm[k], xmat1, k); col3(nrm[3 + k], xmat2, k); }
  for (int j = 0; j < 2; j++)
    for (int k = 0; k < 3; k++) {
      double proj[2], radius[2];
      for (int i = 0; i < 2; i++) {
        const double* sz = i == 0 ? s1 : s2;
        const double* ax = nrm[3 * j + k];
        proj[i] = dot3(xc[i], ax);
        radius[i] = fabs(sz[0] * dot3(nrm[3 * i + 0], ax)) + fabs(sz[1] * dot3(nrm[3 * i + 1], ax)) +
                    fabs(sz[2] * dot3(nrm[3 * i + 2], ax));
      }
      if (radius[0] + radius[1] < fabs(proj[1] - proj[0])) return 0;
    }
  return 1;
}

static int broadphase_filter(const MyoGeom* g1, const MyoGeom* g2, const double* xpos1, const double* xpos2,
                             const double* xmat1, const double* xmat2) {
  if (g1->rbound == 0.0 || g2->rbound == 0.0) return plane_filter(g1->rbound, g2->rbound, xpos1, xpos2, xmat1, xmat2);
  if (!sphere_filter(g1->rbound, g2->rbound, xpos1, xpos2)) return 0;
  if (!aabb_filter(g1->aabb_center, g2->aabb_center, g1->aabb_size, g2->aabb_size, xpos1, xpos2, xmat1, xmat2)) return 0;
  if (!obb_filter(g1->aabb_center, g2->aabb_center, g1->aabb_size, g2->aabb_size, xpos1, xpos2, xmat1, xmat2)) return 0;
  return 1;
}

// ------------------------------------------------------------------------------------------------ primitives
static void closest_segment_point(const double* a, const double* b, const double* pt, double* out) {
  double ab[3], ap[3];
  sub3(ab, b, a); sub3(ap, pt, a);
  double t = dot3(ap, ab) / (dot3(ab, ab) + 1e-6);
  t = clampd(t, 0.0, 1.0);
  out[0] = a[0] + t * ab[0]; out[1] = a[1] + t * ab[1]; out[2] = a[2] + t * ab[2];
}

static double plane_sphere(const double* n, const double* ppos, const double* spos, double r, double* pos) {
  double d[3];
  sub3(d, spos, ppos);
  double dist = dot3(d, n) - r;
  double s = r + 0.5 * dist;
  pos[0] = spos[0] - n[0] * s; pos[1] = spos[1] - n[1] * s; pos[2] = spos[2] - n[2] * s;
  return dist;
}

static double sphere_sphere(const double* p1, double r1, const double* p2, double r2, double* pos, double* n) {
  double dir[3];
  sub3(dir, p2, p1);
  double dist = sqrt(dot3(dir, dir));
  if (dist == 0.0) { n[0] = 1.0; n[1] = 0.0; n[2] = 0.0; }
  else scl3(n, dir, 1.0 / dist);
  dist = dist - (r1 + r2);
  double s = r1 + 0.5 * dist;
  pos[0] = p1[0] + n[0] * s; pos[1] = p1[1] + n[1] * s; pos[2] = p1[2] + n[2] * s;
  return dist;
}

static int emit(MyoContact* out, int* nacon, int naconmax, double dist, const double* pos, const double* normal,
                double curvature, const MyoGeom* G1, const MyoGeom* G2, int g1, int g2) {
  if (!(dist < 0.0)) return 0;
  int cid = (*nacon)++;
  if (cid >= naconmax) return 0;
  MyoContact* c = out + cid;
  c->dist = dist;
  memcpy(c->pos, pos, sizeof(c->pos));
  double nn = sqrt(dot3(normal, normal));
  scl3(c->normal, normal, nn > 0 ? 1.0 / nn : 0.0);
  c->curvature = curvature;
  if (G1->priority == G2->priority) {
    for (int k = 0; k < 3; k++) c->friction[k] = fmax(G1->friction[k], G2->friction[k]);
    c->stiffness = fmax(G1->stiffness, G2->stiffness);
    c->dissipation = fmax(G1->dissipation, G2->dissipation);
    c->transition = fmax(G1->transition, G2->transition);
  } else {
    const MyoGeom* G = G1->priority > G2->priority ? G1 : G2;
    memcpy(c->friction, G->friction, sizeof(c->friction));
    c->stiffness = G->stiffness; c->dissipation = G->dissipation; c->transition = G->transition;
  }
  c->g1 = g1; c->g2 = g2;
  return 1;
}

int myo_collide(int ngeom, const MyoGeom* g, const double* gpos, const double* grot, int npair, const int* pairs,
                int naconmax, int* ncollision, MyoContact* out) {
  int nacon = 0, ncol = 0;
  for (int p = 0; p < npair; p++) {
    int a = pairs[2 * p], b = pairs[2 * p + 1];
    if (!broadphase_filter(g + a, g + b, gpos + 3 * a, gpos + 3 * b, grot + 9 * a, grot + 9 * b)) continue;
    int pid = ncol++;
    if (pid >= naconmax) continue;
    int g1 = a, g2 = b;
    if (g[a].type > g[b].type) { g1 = b; g2 = a; }
    const MyoGeom* G1 = g + g1, *G2 = g + g2;
    const double* p1 = gpos + 3 * g1, *p2 = gpos + 3 * g2, *R1 = grot + 9 * g1, *R2 = grot + 9 * g2;
    int t1 = G1->type, t2 = G2->type;
    double pos[3], n[3];
    if (t1 == MYO_GEOM_PLANE && t2 == MYO_GEOM_SPHERE) {
      col3(n, R1, 1);
      double dist = plane_sphere(n, p1, p2, G2->size[0], pos);
      emit(out, &nacon, naconmax, dist, pos, n, G2->size[0], G1, G2, g1, g2);
    } else if (t1 == MYO_GEOM_PLANE && t2 == MYO_GEOM_CAPSULE) {
      double axis[3], bb[3], seg[3], e[3];
      col3(n, R1, 1); col3(axis, R2, 2);
      // frame tangent (Bolt computes it; only the normal is used by Hunt-Crossley)
      scl3(bb, n, dot3(n, axis)); sub3(bb, axis, bb);
      scl3(seg, axis, G2->size[1]);
      add3(e, p2, seg);
      double d1 = plane_sphere(n, p1, e, G2->size[0], pos);
      emit(out, &nacon, naconmax, d1, pos, n, G2->size[0], G1, G2, g1, g2);
      sub3(e, p2, seg);
      double d2 = plane_sphere(n, p1, e, G2->size[0], pos);
      emit(out, &nacon, naconmax, d2, pos, n, G2->size[0], G1, G2, g1, g2);
    } else if (t1 == MYO_GEOM_SPHERE && t2 == MYO_GEOM_SPHERE) {
      double dist = sphere_sphere(p1, G1->size[0], p2, G2->size[0], pos, n);
      emit(out, &nacon, naconmax, dist, pos, n, sqrt(G1->size[0] * G2->size[0]), G1, G2, g1, g2);
    } else if (t1 == MYO_GEOM_SPHERE && t2 == MYO_GEOM_CAPSULE) {
      double axis[3], seg[3], a0[3], a1[3], pt[3];
      col3(axis, R2, 2);
      scl3(seg, axis, G2->size[1]);
      sub3(a0, p2, seg); add3(a1, p2, seg);
      closest_segment_point(a0, a1, p1, pt);
      double dist = sphere_sphere(p1, G1->size[0], pt, G2->size[0], pos, n);
      emit(out, &nacon, naconmax, dist, pos, n, sqrt(G1->size[0] * G2->size[0]), G1, G2, g1, g2);
    } else if (t1 == MYO_GEOM_CAPSULE && t2 == MYO_GEOM_CAPSULE) {
      double ax1[3], ax2[3], axis1[3], axis2[3], dif[3];
      col3(ax1, R1, 2); col3(ax2, R2, 2);
      scl3(axis1, ax1, G1->size[1]); scl3(axis2, ax2, G2->size[1]);
      sub3(dif, p1, p2);
      double r1 = G1->size[0], r2 = G2->size[0], curv = sqrt(r1 * r2);
      double ma = dot3(axis1, axis1), mb = -dot3(axis1, axis2), mc = dot3(axis2, axis2);
      double u = -dot3(axis1, dif), v = dot3(axis2, dif), det = ma * mc - mb * mb;
      double cd[2] = {INFINITY, INFINITY}, cp[2][3], cn[2][3];
      if (fabs(det) >= MYO_BOLT_MINVAL) {
        double inv = 1.0 / det, x1 = (mc * u - mb * v) * inv, x2 = (ma * v - mb * u) * inv;
        if (x1 > 1.0) { x1 = 1.0; x2 = (v - mb) / mc; }
        else if (x1 < -1.0) { x1 = -1.0; x2 = (v + mb) / mc; }
        if (x2 > 1.0) { x2 = 1.0; x1 = clampd((u - mb) / ma, -1.0, 1.0); }
        else if (x2 < -1.0) { x2 = -1.0; x1 = clampd((u + mb) / ma, -1.0, 1.0); }
        double v1[3], v2[3], t[3];
        scl3(t, axis1, x1); add3(v1, p1, t);
        scl3(t, axis2, x2); add3(v2, p2, t);
        double dist = sphere_sphere(v1, r1, v2, r2, pos, n);
        if (dist <= 0.0) { cd[0] = dist; memcpy(cp[0], pos, 24); memcpy(cn[0], n, 24); }
      } else {
        int cnt = 0;
        double v1[3], v2[3], t[3], x1, x2, dist;
        // x1 = +1
        add3(v1, p1, axis1); x2 = clampd((v - mb) / mc, -1.0, 1.0); scl3(t, axis2, x2); add3(v2, p2, t);
        dist = sphere_sphere(v1, r1, v2, r2, pos, n);
        if (dist <= 0.0) { cd[cnt] = dist; memcpy(cp[cnt], pos, 24); memcpy(cn[cnt], n, 24); cnt++; }
        // x1 = -1
        sub3(v1, p1, axis1); x2 = clampd((v + mb) / mc, -1.0, 1.0); scl3(t, axis2, x2); add3(v2, p2, t);
        dist = sphere_sphere(v1, r1, v2, r2, pos, n);
        if (dist <= 0.0) { cd[cnt] = dist; memcpy(cp[cnt], pos, 24); memcpy(cn[cnt], n, 24); cnt++; }
        if (cnt < 2) {  // x2 = +1
          add3(v2, p2, axis2); x1 = clampd((u - mb) / ma, -1.0, 1.0); scl3(t, axis1, x1); add3(v1, p1, t);
          dist = sphere_sphere(v1, r1, v2, r2, pos, n);
          if (dist <= 0.0) { cd[cnt] = dist; memcpy(cp[cnt], pos, 24); memcpy(cn[cnt], n, 24); cnt++; }
        }
        if (cnt < 2) {  // x2 = -1
          sub3(v2, p2, axis2); x1 = clampd((u + mb) / ma, -1.0, 1.0); scl3(t, axis1, x1); add3(v1, p1, t);
          dist = sphere_sphere(v1, r1, v2, r2, pos, n);
          if (dist <= 0.0) { cd[cnt] = dist; memcpy(cp[cnt], pos, 24); memcpy(cn[cnt], n, 24); }
        }
      }
      for (int i = 0; i < 2; i++)
        if (cd[i] < INFINITY) emit(out, &nacon, naconmax, cd[i], cp[i], cn[i], curv, G1, G2, g1, g2);
    }
  }
  if (ncollision) *ncollision = ncol;
  return nacon < naconmax ? nacon : naconmax;
}

int myo_hc_force(const MyoContact* c, const double v1[3], const double v2[3], double force[3]) {
  double depth = -c->dist;
  force[0] = force[1] = force[2] = 0.0;
  if (depth < 0.0) return 0;
  double k = 0.5 * c->stiffness, cc = c->dissipation, R = c->curvature;
  double fH = (4.0 / 3.0) * k * depth * sqrt(R * k * depth);
  double v[3], vt[3], t[3];
  sub3(v, v1, v2);
  double vn = dot3(v, c->normal);
  scl3(t, c->normal, vn); sub3(vt, v, t);
  double f = fH * (1.0 + 1.5 * cc * vn);
  if (f <= 0.0) return 0;
  scl3(force, c->normal, f);
  double us = c->friction[0], ud = c->friction[1], uv = c->friction[2];
  double vs = sqrt(dot3(vt, vt));
  if (vs > MYO_BOLT_MINVAL) {
    double vrel = vs / c->transition;
    double ff = f * (fmin(vrel, 1.0) * (ud + 2.0 * (us - ud) / (1.0 + vrel * vrel)) + uv * vs);
    for (int i = 0; i < 3; i++) force[i] += ff * vt[i] / vs;
  }
  return 1;
}

// ------------------------------------------------------------------------------------------------ flat (ctypes) API
#define MYO_GEOM_NFLAT 26
#define MYO_CONTACT_NFLAT 17

void myo_geoms_from_flat(int ngeom, const double* f, MyoGeom* g) {
  for (int i = 0; i < ngeom; i++, f += MYO_GEOM_NFLAT) {
    g[i].body = (int)f[0]; g[i].type = (int)f[1];
    memcpy(g[i].size, f + 2, 24); memcpy(g[i].pos, f + 5, 24); memcpy(g[i].quat, f + 8, 32);
    memcpy(g[i].friction, f + 12, 24);
    g[i].stiffness = f[15]; g[i].dissipation = f[16]; g[i].transition = f[17]; g[i].priority = (int)f[18];
    memcpy(g[i].aabb_center, f + 19, 24); memcpy(g[i].aabb_size, f + 22, 24); g[i].rbound = f[25];
  }
}

// contacts for one state; returns nacon (Bolt counter, may exceed naconmax) and writes min(nacon, naconmax) rows
int myo_collide_flat(int ngeom, const double* gflat, const double* bpos, const double* bquat, int npair,
                     const int* pairs, int naconmax, double* cflat, double* gpos, double* grot) {
  MyoGeom g[256];
  MyoContact c[1024];
  if (ngeom > 256 || naconmax > 1024) return -1;
  myo_geoms_from_flat(ngeom, gflat, g);
  myo_geom_poses(ngeom, g, bpos, bquat, gpos, grot);
  int ncol = 0;
  int n = myo_collide(ngeom, g, gpos, grot, npair, pairs, naconmax, &ncol, c);
  for (int i = 0; i < n; i++) {
    double* o = cflat + MYO_CONTACT_NFLAT * i;
    o[0] = c[i].dist; memcpy(o + 1, c[i].pos, 24); memcpy(o + 4, c[i].normal, 24); o[7] = c[i].curvature;
    o[8] = c[i].stiffness; o[9] = c[i].dissipation; o[10] = c[i].transition; memcpy(o + 11, c[i].friction, 24);
    o[14] = c[i].g1; o[15] = c[i].g2; o[16] = ncol;
  }
  return n;
}

int myo_hc_flat(const double* cf, const double* v1, const double* v2, double* force) {
  MyoContact c;
  c.dist = cf[0]; memcpy(c.pos, cf + 1, 24); memcpy(c.normal, cf + 4, 24); c.curvature = cf[7];
  c.stiffness = cf[8]; c.dissipation = cf[9]; c.transition = cf[10]; memcpy(c.friction, cf + 11, 24);
  c.g1 = (int)cf[14]; c.g2 = (int)cf[15];
  return myo_hc_force(&c, v1, v2, force);
}
