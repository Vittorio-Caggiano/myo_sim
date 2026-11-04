"""Hunt-Crossley contact pipeline (broadphase filters, plane/sphere/capsule narrowphase, Hunt-Crossley normal
force + friction) as MuJoCo Warp float32 kernels.

The contact model follows Bolt (https://github.com/willwng/bolt, William Wang et al., Apache-2.0); see NOTICE.

Conventions: body poses are MuJoCo xpos / xquat (w, x, y, z); a body's spatial velocity is given at the body origin as
(angular, linear) in world. Contact wrenches are accumulated about the body origin and, for coupling, converted to
d.xfrc_applied (force, torque about xipos).
"""

from __future__ import annotations

import numpy as np
import warp as wp

from .launch_util import launch as _launch

wp.set_module_options({"enable_backward": False})

PLANE, SPHERE, CAPSULE = 0, 2, 3
MINVAL = 1e-15


@wp.struct
class GeomTable:
    body: wp.array(dtype=int)
    type: wp.array(dtype=int)
    size: wp.array(dtype=wp.vec3)
    pos: wp.array(dtype=wp.vec3)
    quat: wp.array(dtype=wp.vec4)  # w x y z
    friction: wp.array(dtype=wp.vec3)
    stiffness: wp.array(dtype=float)
    dissipation: wp.array(dtype=float)
    transition: wp.array(dtype=float)
    priority: wp.array(dtype=int)
    aabb_center: wp.array(dtype=wp.vec3)
    aabb_size: wp.array(dtype=wp.vec3)
    rbound: wp.array(dtype=float)


@wp.func
def quat_to_mat(q: wp.vec4) -> wp.mat33:
    w = q[0]
    x = q[1]
    y = q[2]
    z = q[3]
    return wp.mat33(
        1.0 - 2.0 * (y * y + z * z),
        2.0 * (x * y - z * w),
        2.0 * (x * z + y * w),
        2.0 * (x * y + z * w),
        1.0 - 2.0 * (x * x + z * z),
        2.0 * (y * z - x * w),
        2.0 * (x * z - y * w),
        2.0 * (y * z + x * w),
        1.0 - 2.0 * (x * x + y * y),
    )


@wp.func
def quat_mul(a: wp.vec4, b: wp.vec4) -> wp.vec4:
    return wp.vec4(
        a[0] * b[0] - a[1] * b[1] - a[2] * b[2] - a[3] * b[3],
        a[0] * b[1] + a[1] * b[0] + a[2] * b[3] - a[3] * b[2],
        a[0] * b[2] - a[1] * b[3] + a[2] * b[0] + a[3] * b[1],
        a[0] * b[3] + a[1] * b[2] - a[2] * b[1] + a[3] * b[0],
    )


@wp.func
def col(R: wp.mat33, c: int) -> wp.vec3:
    return wp.vec3(R[0, c], R[1, c], R[2, c])


# ------------------------------------------------------------------------------------------------ broadphase
@wp.func
def plane_filter(size1: float, size2: float, xpos1: wp.vec3, xpos2: wp.vec3, xmat1: wp.mat33, xmat2: wp.mat33) -> bool:
    if size1 == 0.0:
        return wp.dot(xpos2 - xpos1, col(xmat1, 1)) <= size2
    elif size2 == 0.0:
        return wp.dot(xpos1 - xpos2, col(xmat2, 1)) <= size1
    return True


@wp.func
def sphere_filter(size1: float, size2: float, xpos1: wp.vec3, xpos2: wp.vec3) -> bool:
    bound = size1 + size2
    dif = xpos2 - xpos1
    return wp.dot(dif, dif) <= bound * bound


@wp.func
def aabb_filter(
    c1: wp.vec3, c2: wp.vec3, s1: wp.vec3, s2: wp.vec3, xpos1: wp.vec3, xpos2: wp.vec3, xmat1: wp.mat33, xmat2: wp.mat33
) -> bool:
    center1 = xmat1 @ c1 + xpos1
    center2 = xmat2 @ c2 + xpos2
    max1 = wp.vec3(-1.0e10)
    min1 = wp.vec3(1.0e10)
    max2 = wp.vec3(-1.0e10)
    min2 = wp.vec3(1.0e10)
    sign = wp.vec2(-1.0, 1.0)
    for i in range(2):
        for j in range(2):
            for k in range(2):
                p1 = xmat1 @ wp.vec3(sign[i] * s1[0], sign[j] * s1[1], sign[k] * s1[2])
                p2 = xmat2 @ wp.vec3(sign[i] * s2[0], sign[j] * s2[1], sign[k] * s2[2])
                for a in range(3):
                    if p1[a] > max1[a]:
                        max1[a] = p1[a]
                    if p1[a] < min1[a]:
                        min1[a] = p1[a]
                    if p2[a] > max2[a]:
                        max2[a] = p2[a]
                    if p2[a] < min2[a]:
                        min2[a] = p2[a]
    for a in range(3):
        if center1[a] + max1[a] < center2[a] + min2[a]:
            return False
        if center2[a] + max2[a] < center1[a] + min1[a]:
            return False
    return True


@wp.func
def obb_filter(
    c1: wp.vec3, c2: wp.vec3, s1: wp.vec3, s2: wp.vec3, xpos1: wp.vec3, xpos2: wp.vec3, xmat1: wp.mat33, xmat2: wp.mat33
) -> bool:
    xc0 = xmat1 @ c1 + xpos1
    xc1 = xmat2 @ c2 + xpos2
    for j in range(2):
        for k in range(3):
            if j == 0:
                ax = col(xmat1, k)
            else:
                ax = col(xmat2, k)
            proj0 = wp.dot(xc0, ax)
            proj1 = wp.dot(xc1, ax)
            r0 = (
                wp.abs(s1[0] * wp.dot(col(xmat1, 0), ax))
                + wp.abs(s1[1] * wp.dot(col(xmat1, 1), ax))
                + wp.abs(s1[2] * wp.dot(col(xmat1, 2), ax))
            )
            r1 = (
                wp.abs(s2[0] * wp.dot(col(xmat2, 0), ax))
                + wp.abs(s2[1] * wp.dot(col(xmat2, 1), ax))
                + wp.abs(s2[2] * wp.dot(col(xmat2, 2), ax))
            )
            if r0 + r1 < wp.abs(proj1 - proj0):
                return False
    return True


# ------------------------------------------------------------------------------------------------ primitives
@wp.func
def closest_segment_point(a: wp.vec3, b: wp.vec3, pt: wp.vec3) -> wp.vec3:
    ab = b - a
    t = wp.dot(pt - a, ab) / (wp.dot(ab, ab) + 1e-6)
    return a + wp.clamp(t, 0.0, 1.0) * ab


@wp.func
def sphere_sphere(p1: wp.vec3, r1: float, p2: wp.vec3, r2: float):
    d = p2 - p1
    dist = wp.length(d)
    n = wp.vec3(1.0, 0.0, 0.0)
    if dist != 0.0:
        n = d / dist
    dist = dist - (r1 + r2)
    pos = p1 + n * (r1 + 0.5 * dist)
    return dist, pos, n


@wp.func
def plane_sphere(n: wp.vec3, ppos: wp.vec3, spos: wp.vec3, r: float):
    dist = wp.dot(spos - ppos, n) - r
    pos = spos - n * (r + 0.5 * dist)
    return dist, pos


@wp.kernel
def _geom_poses(
    gt: GeomTable,
    bpos: wp.array2d(dtype=wp.vec3),
    bquat: wp.array2d(dtype=wp.vec4),
    gpos_out: wp.array2d(dtype=wp.vec3),
    grot_out: wp.array2d(dtype=wp.mat33),
):
    worldid, gid = wp.tid()
    b = gt.body[gid]
    qb = bquat[worldid, b]
    gpos_out[worldid, gid] = bpos[worldid, b] + quat_to_mat(qb) @ gt.pos[gid]
    grot_out[worldid, gid] = quat_to_mat(wp.normalize(quat_mul(qb, gt.quat[gid])))


@wp.func
def _emit(
    gt: GeomTable,
    cid: int,
    naconmax: int,
    dist: float,
    pos: wp.vec3,
    normal: wp.vec3,
    curvature: float,
    g1: int,
    g2: int,
    worldid: int,
    c_dist: wp.array2d(dtype=float),
    c_pos: wp.array2d(dtype=wp.vec3),
    c_normal: wp.array2d(dtype=wp.vec3),
    c_curv: wp.array2d(dtype=float),
    c_geom: wp.array2d(dtype=wp.vec2i),
) -> int:
    if not (dist < 0.0):
        return cid
    if cid < naconmax:
        c_dist[worldid, cid] = dist
        c_pos[worldid, cid] = pos
        c_normal[worldid, cid] = wp.normalize(normal)
        c_curv[worldid, cid] = curvature
        c_geom[worldid, cid] = wp.vec2i(g1, g2)
    return cid + 1


@wp.kernel
def _narrowphase(
    gt: GeomTable,
    pairs: wp.array(dtype=wp.vec2i),
    naconmax: int,
    active: wp.array2d(dtype=int),
    nactive: wp.array(dtype=int),
    gpos: wp.array2d(dtype=wp.vec3),
    grot: wp.array2d(dtype=wp.mat33),
    nacon_out: wp.array(dtype=int),
    ncollision_out: wp.array(dtype=int),
    c_dist: wp.array2d(dtype=float),
    c_pos: wp.array2d(dtype=wp.vec3),
    c_normal: wp.array2d(dtype=wp.vec3),
    c_curv: wp.array2d(dtype=float),
    c_geom: wp.array2d(dtype=wp.vec2i),
):
    worldid = wp.tid()
    cid = int(0)
    ncol = int(0)
    # Pair order decides the contact order and with it the float32 force sums, so restore it first.
    for k in range(1, nactive[worldid]):
        x = active[worldid, k]
        j = k - 1
        while j >= 0 and active[worldid, j] > x:
            active[worldid, j + 1] = active[worldid, j]
            j -= 1
        active[worldid, j + 1] = x
    for k in range(nactive[worldid]):
        p = active[worldid, k]
        a = pairs[p][0]
        b = pairs[p][1]
        pid = ncol
        ncol += 1
        if pid >= naconmax:
            continue
        g1 = a
        g2 = b
        if gt.type[a] > gt.type[b]:
            g1 = b
            g2 = a
        p1 = gpos[worldid, g1]
        p2 = gpos[worldid, g2]
        R1 = grot[worldid, g1]
        R2 = grot[worldid, g2]
        t1 = gt.type[g1]
        t2 = gt.type[g2]
        s1 = gt.size[g1]
        s2 = gt.size[g2]
        if t1 == PLANE and t2 == SPHERE:
            n = col(R1, 1)
            dist, pos = plane_sphere(n, p1, p2, s2[0])
            cid = _emit(gt, cid, naconmax, dist, pos, n, s2[0], g1, g2, worldid, c_dist, c_pos, c_normal, c_curv, c_geom)
        elif t1 == PLANE and t2 == CAPSULE:
            n = col(R1, 1)
            seg = col(R2, 2) * s2[1]
            d1, q1 = plane_sphere(n, p1, p2 + seg, s2[0])
            cid = _emit(gt, cid, naconmax, d1, q1, n, s2[0], g1, g2, worldid, c_dist, c_pos, c_normal, c_curv, c_geom)
            d2, q2 = plane_sphere(n, p1, p2 - seg, s2[0])
            cid = _emit(gt, cid, naconmax, d2, q2, n, s2[0], g1, g2, worldid, c_dist, c_pos, c_normal, c_curv, c_geom)
        elif t1 == SPHERE and t2 == SPHERE:
            dist, pos, n = sphere_sphere(p1, s1[0], p2, s2[0])
            cid = _emit(
                gt,
                cid,
                naconmax,
                dist,
                pos,
                n,
                wp.sqrt(s1[0] * s2[0]),
                g1,
                g2,
                worldid,
                c_dist,
                c_pos,
                c_normal,
                c_curv,
                c_geom,
            )
        elif t1 == SPHERE and t2 == CAPSULE:
            seg = col(R2, 2) * s2[1]
            pt = closest_segment_point(p2 - seg, p2 + seg, p1)
            dist, pos, n = sphere_sphere(p1, s1[0], pt, s2[0])
            cid = _emit(
                gt,
                cid,
                naconmax,
                dist,
                pos,
                n,
                wp.sqrt(s1[0] * s2[0]),
                g1,
                g2,
                worldid,
                c_dist,
                c_pos,
                c_normal,
                c_curv,
                c_geom,
            )
        elif t1 == CAPSULE and t2 == CAPSULE:
            axis1 = col(R1, 2) * s1[1]
            axis2 = col(R2, 2) * s2[1]
            dif = p1 - p2
            r1 = s1[0]
            r2 = s2[0]
            curv = wp.sqrt(r1 * r2)
            ma = wp.dot(axis1, axis1)
            mb = -wp.dot(axis1, axis2)
            mc = wp.dot(axis2, axis2)
            u = -wp.dot(axis1, dif)
            v = wp.dot(axis2, dif)
            det = ma * mc - mb * mb
            if wp.abs(det) >= MINVAL:
                inv = 1.0 / det
                x1 = (mc * u - mb * v) * inv
                x2 = (ma * v - mb * u) * inv
                if x1 > 1.0:
                    x1 = 1.0
                    x2 = (v - mb) / mc
                elif x1 < -1.0:
                    x1 = -1.0
                    x2 = (v + mb) / mc
                if x2 > 1.0:
                    x2 = 1.0
                    x1 = wp.clamp((u - mb) / ma, -1.0, 1.0)
                elif x2 < -1.0:
                    x2 = -1.0
                    x1 = wp.clamp((u + mb) / ma, -1.0, 1.0)
                dist, pos, n = sphere_sphere(p1 + axis1 * x1, r1, p2 + axis2 * x2, r2)
                if dist <= 0.0:
                    cid = _emit(gt, cid, naconmax, dist, pos, n, curv, g1, g2, worldid, c_dist, c_pos, c_normal, c_curv, c_geom)
            else:
                cnt = int(0)
                va = wp.vec3(0.0)
                vb = wp.vec3(0.0)
                for e in range(4):
                    if e >= 2 and cnt >= 2:
                        continue
                    if e == 0:
                        va = p1 + axis1
                        vb = p2 + axis2 * wp.clamp((v - mb) / mc, -1.0, 1.0)
                    elif e == 1:
                        va = p1 - axis1
                        vb = p2 + axis2 * wp.clamp((v + mb) / mc, -1.0, 1.0)
                    elif e == 2:
                        vb = p2 + axis2
                        va = p1 + axis1 * wp.clamp((u - mb) / ma, -1.0, 1.0)
                    else:
                        vb = p2 - axis2
                        va = p1 + axis1 * wp.clamp((u + mb) / ma, -1.0, 1.0)
                    dist, pos, n = sphere_sphere(va, r1, vb, r2)
                    if dist <= 0.0:
                        cnt += 1
                        cid = _emit(
                            gt, cid, naconmax, dist, pos, n, curv, g1, g2, worldid, c_dist, c_pos, c_normal, c_curv, c_geom
                        )
    nacon_out[worldid] = wp.min(cid, naconmax)
    ncollision_out[worldid] = ncol


@wp.kernel
def _broadphase(
    gt: GeomTable,
    pairs: wp.array(dtype=wp.vec2i),
    gpos: wp.array2d(dtype=wp.vec3),
    grot: wp.array2d(dtype=wp.mat33),
    active_out: wp.array2d(dtype=int),
    nactive_out: wp.array(dtype=int),
):
    """One thread per world and pair: appends the pairs that pass the filters to the world's active list (unordered)."""
    worldid, p = wp.tid()
    a = pairs[p][0]
    b = pairs[p][1]
    xa = gpos[worldid, a]
    xb = gpos[worldid, b]
    Ra = grot[worldid, a]
    Rb = grot[worldid, b]
    ok = bool(True)
    if gt.rbound[a] == 0.0 or gt.rbound[b] == 0.0:
        ok = plane_filter(gt.rbound[a], gt.rbound[b], xa, xb, Ra, Rb)
    else:
        if not sphere_filter(gt.rbound[a], gt.rbound[b], xa, xb):
            ok = False
        elif not aabb_filter(gt.aabb_center[a], gt.aabb_center[b], gt.aabb_size[a], gt.aabb_size[b], xa, xb, Ra, Rb):
            ok = False
        elif not obb_filter(gt.aabb_center[a], gt.aabb_center[b], gt.aabb_size[a], gt.aabb_size[b], xa, xb, Ra, Rb):
            ok = False
    if ok:
        active_out[worldid, wp.atomic_add(nactive_out, worldid, 1)] = p


@wp.func
def _param_max_or_priority(v1: float, v2: float, p1: int, p2: int) -> float:
    if p1 == p2:
        return wp.max(v1, v2)
    elif p1 > p2:
        return v1
    return v2


@wp.kernel
def _hunt_crossley(
    gt: GeomTable,
    nacon: wp.array(dtype=int),
    c_dist: wp.array2d(dtype=float),
    c_pos: wp.array2d(dtype=wp.vec3),
    c_normal: wp.array2d(dtype=wp.vec3),
    c_curv: wp.array2d(dtype=float),
    c_geom: wp.array2d(dtype=wp.vec2i),
    bpos: wp.array2d(dtype=wp.vec3),
    bvel: wp.array2d(dtype=wp.spatial_vector),
    wrench_out: wp.array2d(dtype=wp.spatial_vector),
    grf_out: wp.array(dtype=wp.vec3),
):
    worldid = wp.tid()
    grf = wp.vec3(0.0)
    for cid in range(nacon[worldid]):
        depth = -c_dist[worldid, cid]
        if depth < 0.0:
            continue
        g1 = c_geom[worldid, cid][0]
        g2 = c_geom[worldid, cid][1]
        b1 = gt.body[g1]
        b2 = gt.body[g2]
        pr1 = gt.priority[g1]
        pr2 = gt.priority[g2]
        stiffness = _param_max_or_priority(gt.stiffness[g1], gt.stiffness[g2], pr1, pr2)
        cdiss = _param_max_or_priority(gt.dissipation[g1], gt.dissipation[g2], pr1, pr2)
        trans = _param_max_or_priority(gt.transition[g1], gt.transition[g2], pr1, pr2)
        us = _param_max_or_priority(gt.friction[g1][0], gt.friction[g2][0], pr1, pr2)
        ud = _param_max_or_priority(gt.friction[g1][1], gt.friction[g2][1], pr1, pr2)
        uv = _param_max_or_priority(gt.friction[g1][2], gt.friction[g2][2], pr1, pr2)
        loc = c_pos[worldid, cid]
        normal = c_normal[worldid, cid]
        k = 0.5 * stiffness
        fH = (4.0 / 3.0) * k * depth * wp.sqrt(c_curv[worldid, cid] * k * depth)
        r1 = loc - bpos[worldid, b1]
        r2 = loc - bpos[worldid, b2]
        V1 = bvel[worldid, b1]
        V2 = bvel[worldid, b2]
        v1 = wp.spatial_bottom(V1) + wp.cross(wp.spatial_top(V1), r1)
        v2 = wp.spatial_bottom(V2) + wp.cross(wp.spatial_top(V2), r2)
        vr = v1 - v2
        vn = wp.dot(vr, normal)
        vt = vr - vn * normal
        f = fH * (1.0 + 1.5 * cdiss * vn)
        if f <= 0.0:
            continue
        force = f * normal
        vs = wp.length(vt)
        if vs > MINVAL:
            vrel = vs / trans
            ff = f * (wp.min(vrel, 1.0) * (ud + 2.0 * (us - ud) / (1.0 + vrel * vrel)) + uv * vs)
            force += ff * vt / vs
        wrench_out[worldid, b1] = wrench_out[worldid, b1] + wp.spatial_vector(wp.cross(r1, -force), -force)
        wrench_out[worldid, b2] = wrench_out[worldid, b2] + wp.spatial_vector(wp.cross(r2, force), force)
        if b1 == 0:
            grf += force
        elif b2 == 0:
            grf -= force
    grf_out[worldid] = grf


@wp.kernel
def _mj_body_state(
    body_rootid: wp.array(dtype=int),
    xpos: wp.array2d(dtype=wp.vec3),
    xquat: wp.array2d(dtype=wp.quat),
    cvel: wp.array2d(dtype=wp.spatial_vector),
    subtree_com: wp.array2d(dtype=wp.vec3),
    bpos_out: wp.array2d(dtype=wp.vec3),
    bquat_out: wp.array2d(dtype=wp.vec4),
    bvel_out: wp.array2d(dtype=wp.spatial_vector),
):
    worldid, bodyid = wp.tid()
    q = xquat[worldid, bodyid]
    p = xpos[worldid, bodyid]
    bpos_out[worldid, bodyid] = p
    bquat_out[worldid, bodyid] = wp.vec4(q[0], q[1], q[2], q[3])
    cv = cvel[worldid, bodyid]
    w = wp.spatial_top(cv)
    vlin = wp.spatial_bottom(cv) + wp.cross(w, p - subtree_com[worldid, body_rootid[bodyid]])
    bvel_out[worldid, bodyid] = wp.spatial_vector(w, vlin)


@wp.kernel
def _wrench_to_xfrc(
    xpos: wp.array2d(dtype=wp.vec3),
    xipos: wp.array2d(dtype=wp.vec3),
    wrench: wp.array2d(dtype=wp.spatial_vector),
    xfrc_out: wp.array2d(dtype=wp.spatial_vector),
):
    worldid, bodyid = wp.tid()
    wr = wrench[worldid, bodyid]
    torque_o = wp.spatial_top(wr)
    force = wp.spatial_bottom(wr)
    # torque about xipos: tau_P = tau_O + (O - P) x F
    torque = torque_o + wp.cross(xpos[worldid, bodyid] - xipos[worldid, bodyid], force)
    xfrc_out[worldid, bodyid] = xfrc_out[worldid, bodyid] + wp.spatial_vector(force, torque)


class ContactSystem:
    """geom_flat: rows in the c/myo_contact.c flat layout with simulator body ids; pairs: (npair, 2) Bolt order."""

    def __init__(self, geom_flat, pairs, nworld, nbody, naconmax=512, device=None):
        g = np.asarray(geom_flat, np.float64)
        self.nworld, self.nbody, self.ngeom, self.naconmax = nworld, nbody, g.shape[0], naconmax
        gt = GeomTable()
        gt.body = wp.array(g[:, 0].astype(np.int32), dtype=int, device=device)
        gt.type = wp.array(g[:, 1].astype(np.int32), dtype=int, device=device)
        gt.size = wp.array(g[:, 2:5].astype(np.float32), dtype=wp.vec3, device=device)
        gt.pos = wp.array(g[:, 5:8].astype(np.float32), dtype=wp.vec3, device=device)
        gt.quat = wp.array(g[:, 8:12].astype(np.float32), dtype=wp.vec4, device=device)
        gt.friction = wp.array(g[:, 12:15].astype(np.float32), dtype=wp.vec3, device=device)
        gt.stiffness = wp.array(g[:, 15].astype(np.float32), dtype=float, device=device)
        gt.dissipation = wp.array(g[:, 16].astype(np.float32), dtype=float, device=device)
        gt.transition = wp.array(g[:, 17].astype(np.float32), dtype=float, device=device)
        gt.priority = wp.array(g[:, 18].astype(np.int32), dtype=int, device=device)
        gt.aabb_center = wp.array(g[:, 19:22].astype(np.float32), dtype=wp.vec3, device=device)
        gt.aabb_size = wp.array(g[:, 22:25].astype(np.float32), dtype=wp.vec3, device=device)
        gt.rbound = wp.array(g[:, 25].astype(np.float32), dtype=float, device=device)
        self.gt = gt
        self.pairs = wp.array(np.asarray(pairs, np.int32).reshape(-1, 2), dtype=wp.vec2i, device=device)
        z = lambda shape, dt: wp.zeros(shape, dtype=dt, device=device)  # noqa
        self.gpos = z((nworld, self.ngeom), wp.vec3)
        self.grot = z((nworld, self.ngeom), wp.mat33)
        npair = self.pairs.shape[0]
        self.active = z((nworld, npair), int)
        self.nactive = z(nworld, int)
        self.nacon = z(nworld, int)
        self.ncollision = z(nworld, int)
        self.c_dist = z((nworld, naconmax), float)
        self.c_pos = z((nworld, naconmax), wp.vec3)
        self.c_normal = z((nworld, naconmax), wp.vec3)
        self.c_curv = z((nworld, naconmax), float)
        self.c_geom = z((nworld, naconmax), wp.vec2i)
        self.bpos = z((nworld, nbody), wp.vec3)
        self.bquat = z((nworld, nbody), wp.vec4)
        self.bvel = z((nworld, nbody), wp.spatial_vector)
        self.wrench = z((nworld, nbody), wp.spatial_vector)
        self.grf = z(nworld, wp.vec3)
        self.body_rootid = None

    def run(self):
        """Contacts and wrenches from self.bpos / self.bquat / self.bvel."""
        _launch(
            _geom_poses, dim=(self.nworld, self.ngeom), inputs=[self.gt, self.bpos, self.bquat], outputs=[self.gpos, self.grot]
        )
        npair = self.pairs.shape[0]
        self.nactive.zero_()
        _launch(
            _broadphase,
            dim=(self.nworld, npair),
            inputs=[self.gt, self.pairs, self.gpos, self.grot],
            outputs=[self.active, self.nactive],
        )
        _launch(
            _narrowphase,
            dim=self.nworld,
            inputs=[self.gt, self.pairs, self.naconmax, self.active, self.nactive, self.gpos, self.grot],
            outputs=[self.nacon, self.ncollision, self.c_dist, self.c_pos, self.c_normal, self.c_curv, self.c_geom],
        )
        self.wrench.zero_()
        _launch(
            _hunt_crossley,
            dim=self.nworld,
            inputs=[
                self.gt,
                self.nacon,
                self.c_dist,
                self.c_pos,
                self.c_normal,
                self.c_curv,
                self.c_geom,
                self.bpos,
                self.bvel,
            ],
            outputs=[self.wrench, self.grf],
        )

    def compute(self, m, d):
        """From mujoco_warp data (after kinematics and com velocities)."""
        _launch(
            _mj_body_state,
            dim=(self.nworld, self.nbody),
            inputs=[m.body_rootid, d.xpos, d.xquat, d.cvel, d.subtree_com],
            outputs=[self.bpos, self.bquat, self.bvel],
        )
        self.run()

    def apply(self, d):
        """Adds the contact wrenches to d.xfrc_applied (caller zeroes it once per evaluation)."""
        _launch(_wrench_to_xfrc, dim=(self.nworld, self.nbody), inputs=[d.xpos, d.xipos, self.wrench], outputs=[d.xfrc_applied])
