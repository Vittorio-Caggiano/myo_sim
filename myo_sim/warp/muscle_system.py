"""MuJoCo Warp muscle contraction dynamics (float32), De Groote-Fregly 2016 formulation.

The formulation (DGF curves, pennation, damped/undamped/rigid contraction, reset equilibrium, clamps and
float32 operation order) follows Bolt (https://github.com/willwng/bolt, William Wang et al., Apache-2.0);
see NOTICE. Curve constants follow De Groote et al. (2016) and OpenSim's Apache-2.0 DeGrooteFregly2016Muscle.

* batched over (nworld, nmuscle); MuJoCo-native curves reuse mujoco_warp's own util_misc kernels
* coupling to mujoco_warp: muscle forces are written to d.qfrc_applied before mjw.step, muscle states are
  advanced after it (same ordering as a MuJoCo passive-plugin muscle actuator).
"""

from __future__ import annotations

import numpy as np
import warp as wp
from mujoco_warp._src import util_misc as mjw_util
from mujoco_warp._src.types import vec10

from . import consts as K

wp.set_module_options({"enable_backward": False})

NPARAM = 21
NOUT = 12
NSTATE = 10
MAXD = 12
MAXO = 9

(
    P_F0,
    P_LOPT,
    P_LTS,
    P_ALPHA0,
    P_VMAX,
    P_DAMPING,
    P_E0,
    P_TSTRAIN,
    P_WSCALE,
    P_LMIN,
    P_LMAX,
    P_TAU_ACT,
    P_TAU_DEACT,
    P_ACT_SMOOTH,
    P_AMIN,
    P_AMAX,
    P_SPEC_TENSION,
    P_DENSITY,
    P_SLOW_TWITCH,
    P_RIGID,
    P_ACT_MODEL,
) = range(21)

B11, B21, B31, B41 = K.B11, K.B21, K.B31, K.B41
B12, B22, B32, B42 = K.B12, K.B22, K.B32, K.B42
B13, B23, B33, B43 = K.B13, K.B23, K.B33, K.B43
C1, C2, C3 = K.C1, K.C2, K.C3
D1, D2, D3, D4 = K.D1, K.D2, K.D3, K.D4
KPE = K.KPE
MAX_PEN = K.MAX_PENNATION_ANGLE
SIG_REAL = K.SIG_REAL
PEN_EPS = K.PENNATION_EPS
MIN_NTF = K.MIN_NORM_TENDON_FORCE
MAX_NTF = K.MAX_NORM_TENDON_FORCE
EQ_ITERS = K.EQ_BISECT_ITERS
EQ_TOL = K.EQ_TOL
DN_ITERS = K.DAMPED_NEWTON_ITERS
DN_TOL_ABS = K.DAMPED_TOL_ABS
DN_TOL_REL = K.DAMPED_TOL_REL


class vec21(wp.types.vector(length=NPARAM, dtype=float)):
    pass


class vec12(wp.types.vector(length=NOUT, dtype=float)):
    pass


class vecd(wp.types.vector(length=MAXD, dtype=float)):
    pass


class vecr(wp.types.vector(length=MAXD + 1, dtype=float)):
    pass


class matpw(wp.types.matrix(shape=(MAXD, MAXO + 1), dtype=float)):
    pass


# --------------------------------------------------------------------------------------------- curves
@wp.func
def _gauss(x: float, b1: float, b2: float, b3: float, b4: float) -> float:
    den = b3 + b4 * x
    return b1 * wp.exp(-0.5 * (x - b2) * (x - b2) / (den * den))


@wp.func
def fl_active(ln: float, ws: float) -> float:
    x = (ln - 1.0) / ws + 1.0
    return _gauss(x, B11, B21, B31, B41) + _gauss(x, B12, B22, B32, B42) + _gauss(x, B13, B23, B33, B43)


@wp.func
def fv(v: float) -> float:
    t = D2 * v + D3
    s = wp.sqrt(t * t + 1.0)
    arg = wp.where(t >= 0.0, t + s, 1.0 / (s - t))
    return D1 * wp.log(arg) + D4


@wp.func
def fv_deriv(v: float) -> float:
    t = D2 * v + D3
    return (D1 * D2) / wp.sqrt(t * t + 1.0)


@wp.func
def fv_inv(f: float) -> float:
    return (wp.sinh((f - D4) / D1) - D3) / D2


@wp.func
def fp(ln: float, lmin: float, e0: float) -> float:
    offset = wp.exp(KPE * (lmin - 1.0) / e0)
    return (wp.exp(KPE * (ln - 1.0) / e0) - offset) / (wp.exp(KPE) - offset)


@wp.func
def tendon_kT(strain: float) -> float:
    return wp.log((1.0 + C3) / C1) / (1.0 + strain - C2)


@wp.func
def ft(ltn: float, strain: float, clamped: bool) -> float:
    f = C1 * wp.exp(tendon_kT(strain) * (ltn - C2)) - C3
    return wp.where(clamped, wp.clamp(f, MIN_NTF, MAX_NTF), f)


@wp.func
def ft_inv(f: float, strain: float) -> float:
    return wp.log((f + C3) / C1) / tendon_kT(strain) + C2


# --------------------------------------------------------------------------------------------- activation
@wp.func
def act_dot(a: float, u: float, p: vec21) -> float:
    ta = p[P_TAU_ACT]
    td = p[P_TAU_DEACT]
    b = p[P_ACT_SMOOTH]
    if p[P_ACT_MODEL] == 0.0:
        tcf = 0.5 + 1.5 * a
        f = 0.5 * wp.tanh(b * (u - a))
        tc = (1.0 / (ta * tcf)) * (f + 0.5) + (tcf / td) * (-f + 0.5)
        return tc * (u - a)
    tau = wp.where(u > a, ta * (0.5 + 1.5 * a), td / (0.5 + 1.5 * a))
    return (u - a) / tau


# --------------------------------------------------------------------------------------------- contraction
@wp.func
def pennation_angle(ln: float, p: vec21) -> float:
    lopt = p[P_LOPT]
    alpha0 = p[P_ALPHA0]
    if not (alpha0 > PEN_EPS):
        return 0.0
    if not (ln > p[P_LMIN]):
        return MAX_PEN
    h = lopt * wp.sin(alpha0)
    sin_phi = h / (ln * lopt)
    return wp.where(sin_phi < wp.sin(MAX_PEN), wp.asin(wp.min(sin_phi, 1.0)), MAX_PEN)


@wp.func
def rigid_fiber_length(path_len: float, p: vec21) -> float:
    lopt = p[P_LOPT]
    at = path_len - p[P_LTS]
    min_at = p[P_LMIN] * lopt * wp.cos(MAX_PEN)
    h = lopt * wp.sin(p[P_ALPHA0])
    lm = wp.where(at >= min_at, wp.sqrt(h * h + at * at), min_at)
    return wp.clamp(lm, p[P_LMIN] * lopt, p[P_LMAX] * lopt)


@wp.func
def damped_fiber_velocity(F0: float, a: float, fal: float, fpe: float, fse: float, beta: float, cphi: float) -> wp.vec2:
    fv0 = (fse / wp.max(cphi, 0.01) - fpe) / (wp.max(a, 0.01) * wp.max(fal, 0.01))
    v = wp.clamp(fv_inv(fv0), -1.0, 1.0)
    tol = wp.max(DN_TOL_REL * F0, DN_TOL_ABS)
    for _i in range(DN_ITERS):
        fvv = fv(v)
        dfv = fv_deriv(v)
        err = F0 * (a * fal * fvv + fpe + beta * v) * cphi - fse * F0
        derr = F0 * (a * fal * dfv + beta) * cphi
        if wp.abs(err) > tol and wp.abs(derr) > SIG_REAL:
            v = v - err / derr
    return wp.vec2(v, fv(v))


@wp.func
def contraction(path_len: float, path_vel: float, a: float, ln_state: float, p: vec21) -> vec12:
    F0 = p[P_F0]
    lopt = p[P_LOPT]
    lts = p[P_LTS]
    beta = p[P_DAMPING]
    lmin = p[P_LMIN]
    rigid = p[P_RIGID] > 0.5

    ln = ln_state
    if rigid:
        ln = rigid_fiber_length(path_len, p) / lopt
    lm = ln * lopt
    phi = pennation_angle(ln, p)
    cphi = wp.cos(phi)
    lt = path_len - lm * cphi
    ltn = lt / lts
    fpe = fp(ln, lmin, p[P_E0])
    fal = fl_active(ln, p[P_WSCALE])
    fse = ft(ltn, p[P_TSTRAIN], True)
    vmax_ms = p[P_VMAX] * lopt

    vn = float(0.0)
    fvm = float(1.0)
    if rigid:
        if lt < lts - SIG_REAL:
            vn = 0.0
            fvm = 1.0
        else:
            vn = (path_vel * cphi) / vmax_ms
            fvm = fv(vn)
    elif beta > 0.0:
        r = damped_fiber_velocity(F0, a, fal, fpe, fse, beta, cphi)
        vn = r[0]
        fvm = r[1]
    else:
        fvm = (fse / cphi - fpe) / (a * fal)
        vn = fv_inv(fvm)
    vm = vn * vmax_ms
    clamped = (ln <= lmin and vn < 0.0) or (ln < lmin)
    if clamped:
        vn = 0.0
        vm = 0.0
        fvm = 1.0

    aFm = float(0.0)
    fmAT = float(0.0)
    if not clamped:
        aFm = F0 * a * fal * fvm
        fm = aFm + F0 * fpe + F0 * beta * vn
        if rigid and fm < 0.0:
            fm = 0.0
        fmAT = fm * cphi
    fse_out = wp.where(rigid, fmAT / F0, fse)
    out = vec12(0.0)
    out[0] = fse_out * F0
    out[1] = ln
    out[2] = vn
    out[3] = vm
    out[4] = aFm
    out[5] = fal
    out[6] = fvm
    out[7] = fpe
    out[8] = phi
    out[9] = wp.where(rigid, 0.0, vm / lopt)
    out[10] = lt
    out[11] = wp.where(clamped, 1.0, 0.0)
    return out


# --------------------------------------------------------------------------------------------- equilibrium
@wp.func
def eq_residual(ftn: float, L: float, V: float, a: float, p: vec21) -> wp.vec2:
    F0 = p[P_F0]
    lopt = p[P_LOPT]
    ltn = ft_inv(ftn, p[P_TSTRAIN])
    lt = ltn * p[P_LTS]
    h = lopt * wp.sin(p[P_ALPHA0])
    lm_at = L - lt
    lm = wp.sqrt(lm_at * lm_at + h * h)
    ln = lm / lopt
    cphi = lm_at / lm
    phi = wp.asin(h / lm)
    if phi > wp.acos(0.1):
        cphi = 0.1
    vn = (V * cphi) / (p[P_VMAX] * lopt)
    fiber = F0 * a * fl_active(ln, p[P_WSCALE]) * fv(vn) + F0 * (fp(ln, p[P_LMIN], p[P_E0]) + p[P_DAMPING] * vn)
    return wp.vec2(ftn - fiber * cphi / F0, lm)


@wp.func
def equilibrium(L: float, V: float, a: float, p: vec21) -> float:
    if p[P_RIGID] > 0.5:
        return 0.0
    lo = float(MIN_NTF)
    hi = float(MAX_NTF)
    mid = 0.5 * (lo + hi)
    rl = eq_residual(lo, L, V, a, p)
    rh = eq_residual(hi, L, V, a, p)
    rm = eq_residual(mid, L, V, a, p)
    r_lo = rl[0]
    r_mid = rm[0]
    best_r = rh[0]
    best_lm = rh[1]
    if wp.abs(rl[0]) < wp.abs(rh[0]):
        best_r = rl[0]
        best_lm = rl[1]
    active = bool(True)
    for _i in range(EQ_ITERS):
        if wp.abs(best_r) < EQ_TOL or 0.5 * (hi - lo) < EQ_TOL:
            active = False
        if active:
            if r_lo * r_mid > 0.0:
                lo = mid
                r_lo = r_mid
            else:
                hi = mid
            mid = 0.5 * (lo + hi)
            rn = eq_residual(mid, L, V, a, p)
            r_mid = rn[0]
            if wp.abs(rn[0]) < wp.abs(best_r):
                best_r = rn[0]
                best_lm = rn[1]
    return wp.clamp(best_lm / p[P_LOPT], p[P_LMIN], p[P_LMAX])


# --------------------------------------------------------------------------------------------- metabolics
@wp.func
def metabolics(a_in: float, u_in: float, o: vec12, p: vec21, flags: int, aerobic: float, effort: float) -> float:
    F0 = p[P_F0]
    lopt = p[P_LOPT]
    vmax = p[P_VMAX]
    mass = (F0 / p[P_SPEC_TENSION]) * p[P_DENSITY] * lopt
    s = p[P_SLOW_TWITCH]
    a = effort * a_in
    u = effort * u_in
    fa = wp.max(effort * o[4], 0.0)
    ln = o[1]
    vm = o[3]
    vnorm = vm / lopt
    fiso = o[5]
    bolt = (flags & 16) != 0
    neg_work = (flags & 2) != 0
    A = wp.where(u > a, u, 0.5 * (u + a))
    if (flags & 1) != 0:
        us = s * wp.sin(0.5 * wp.pi * u)
        uf = (1.0 - s) * (1.0 - wp.cos(0.5 * wp.pi * u))
        if u == 0.0:
            s = 1.0
        else:
            s = us / (us + uf)
    am_u = 128.0 * (1.0 - s) + 25.0
    AM = float(0.0)
    if ln <= 1.0:
        AM = aerobic * wp.pow(A, 0.6) * am_u
    else:
        AM = wp.where(bolt, 1.0, aerobic) * wp.pow(A, 0.6) * (0.4 * am_u + 0.6 * am_u * fiso)
    x = wp.where(bolt, ln, vnorm)
    a_ft = 153.0 / vmax
    a_st = 100.0 / (vmax / 2.5)
    S = float(0.0)
    if x <= 0.0:
        st = wp.min(-a_st * x, 100.0)
        S = aerobic * wp.pow(A, 2.0) * (st * s - a_ft * x * (1.0 - s))
    else:
        S = aerobic * A * (wp.where(neg_work, 4.0, 0.3) * a_st * x)
    if ln > 1.0:
        S = S * fiso
    W = float(0.0)
    if neg_work or vm <= 0.0:
        W = -fa * vm
    W = W / mass
    if (flags & 4) != 0:
        e = AM + S + W
        if e < 0.0:
            S = S - e
    heat = AM + S
    if (flags & 8) != 0 and heat < 1.0:
        heat = 1.0
    return (heat + W) * mass


# --------------------------------------------------------------------------------------------- polynomial path
@wp.func
def poly_eval(
    dim: int,
    order: int,
    nterm: int,
    exps: wp.array(dtype=int),
    exp_adr: int,
    coef: wp.array(dtype=float),
    coef_adr: int,
    q: vecd,
) -> vecr:
    pw = matpw(0.0)
    res = vecr(0.0)
    for j in range(dim):
        pw[j, 0] = 1.0
        for e in range(1, order + 1):
            pw[j, e] = pw[j, e - 1] * q[j]
    L = float(0.0)
    for k in range(nterm):
        base = exp_adr + k * dim
        c = coef[coef_adr + k]
        term = c
        for j in range(dim):
            ej = exps[base + j]
            if ej != 0:
                term = term * pw[j, ej]
        L = L + term
        for j in range(dim):
            ej = exps[base + j]
            if ej == 0:
                continue
            t = c * float(ej)
            for i in range(dim):
                ei = exps[base + i]
                if i == j:
                    if ei - 1 > 0:
                        t = t * pw[i, ei - 1]
                elif ei != 0:
                    t = t * pw[i, ei]
            res[j + 1] = res[j + 1] + t
    res[0] = L
    return res


# --------------------------------------------------------------------------------------------- system kernels
@wp.kernel
def _muscle_compute(
    # model
    params: wp.array(dtype=vec21),
    family: wp.array(dtype=int),
    ctrl_id: wp.array(dtype=int),
    dim: wp.array(dtype=int),
    order: wp.array(dtype=int),
    nterm: wp.array(dtype=int),
    exps: wp.array(dtype=int),
    exp_adr: wp.array(dtype=int),
    coef: wp.array(dtype=float),
    coef_adr: wp.array(dtype=int),
    qposadr: wp.array2d(dtype=int),
    dofadr: wp.array2d(dtype=int),
    gainprm: wp.array(dtype=vec10),
    lengthrange: wp.array(dtype=wp.vec2),
    acc0: wp.array(dtype=float),
    dynprm: wp.array(dtype=vec10),
    met_flags: int,
    aerobic: float,
    effort: float,
    pt_tendon: wp.array(dtype=int),
    # data in
    qpos: wp.array2d(dtype=float),
    qvel: wp.array2d(dtype=float),
    ctrl: wp.array2d(dtype=float),
    ten_length: wp.array2d(dtype=float),
    ten_velocity: wp.array2d(dtype=float),
    # data out
    state: wp.array3d(dtype=float),
    grad_out: wp.array3d(dtype=float),
):
    worldid, i = wp.tid()
    d = dim[i]
    q = vecd(0.0)
    for j in range(d):
        q[j] = qpos[worldid, qposadr[i, j]]
    r = poly_eval(d, order[i], nterm[i], exps, exp_adr[i], coef, coef_adr[i], q)
    L = r[0]
    V = float(0.0)
    for j in range(d):
        V = V + r[j + 1] * qvel[worldid, dofadr[i, j]]
        grad_out[worldid, i, j] = r[j + 1]
    if pt_tendon[i] >= 0:  # Bolt as released: point-path length/velocity overwrite the polynomial ones
        L = ten_length[worldid, pt_tendon[i]]
        V = ten_velocity[worldid, pt_tendon[i]]
    u = wp.clamp(ctrl[worldid, ctrl_id[i]], 0.0, 1.0)
    act = state[worldid, i, 0]
    p = params[i]
    F = float(0.0)
    if family[i] == 0:
        o = contraction(L, V, act, state[worldid, i, 1], p)
        F = o[0]
        state[worldid, i, 5] = act_dot(act, u, p)
        state[worldid, i, 6] = o[9]
        state[worldid, i, 7] = metabolics(act, u, o, p, met_flags, aerobic, effort)
        state[worldid, i, 8] = o[3]
        state[worldid, i, 9] = o[1]
    else:
        gain = mjw_util.muscle_gain(L, V, lengthrange[i], acc0[i], gainprm[i])
        bias = mjw_util.muscle_bias(L, lengthrange[i], acc0[i], gainprm[i])
        F = -(gain * act + bias)
        state[worldid, i, 5] = mjw_util.muscle_dynamics(u, act, dynprm[i])
        state[worldid, i, 6] = 0.0
        state[worldid, i, 7] = 0.0
        state[worldid, i, 8] = 0.0
        state[worldid, i, 9] = 0.0
    state[worldid, i, 2] = F
    state[worldid, i, 3] = L
    state[worldid, i, 4] = V


@wp.kernel
def _accumulate_qfrc(
    nmuscle: int,
    dim: wp.array(dtype=int),
    dofadr: wp.array2d(dtype=int),
    state: wp.array3d(dtype=float),
    grad: wp.array3d(dtype=float),
    qfrc: wp.array2d(dtype=float),
):
    worldid = wp.tid()
    for i in range(nmuscle):
        F = state[worldid, i, 2]
        for j in range(dim[i]):
            qfrc[worldid, dofadr[i, j]] = qfrc[worldid, dofadr[i, j]] - F * grad[worldid, i, j]


@wp.kernel
def _accumulate_tendon_qfrc(
    pt_muscles: wp.array(dtype=int),
    pt_tendon: wp.array(dtype=int),
    ten_J_rownnz: wp.array(dtype=int),
    ten_J_rowadr: wp.array(dtype=int),
    ten_J_colind: wp.array(dtype=int),
    ten_J: wp.array2d(dtype=float),
    state: wp.array3d(dtype=float),
    qfrc: wp.array2d(dtype=float),
):
    worldid, k = wp.tid()
    i = pt_muscles[k]
    t = pt_tendon[i]
    F = state[worldid, i, 2]
    adr = ten_J_rowadr[t]
    for j in range(ten_J_rownnz[t]):
        c = ten_J_colind[adr + j]
        wp.atomic_add(qfrc, worldid, c, -F * ten_J[worldid, adr + j])


@wp.func
def step_function(x: float, t0: float, t1: float, v0: float, v1: float) -> float:
    if x <= t0:
        return v0
    if x >= t1:
        return v1
    t = (x - t0) / (t1 - t0)
    return v0 + t * t * (3.0 - 2.0 * t) * (v1 - v0)


@wp.kernel
def _limits(
    lim_adr: wp.array2d(dtype=int),
    lim_prm: wp.array2d(dtype=float),
    nlimit: int,
    qpos: wp.array2d(dtype=float),
    qvel: wp.array2d(dtype=float),
    qfrc: wp.array2d(dtype=float),
):
    worldid = wp.tid()
    for i in range(nlimit):
        q = qpos[worldid, lim_adr[i, 0]]
        qd = qvel[worldid, lim_adr[i, 1]]
        q_low = lim_prm[i, 0]
        q_up = lim_prm[i, 1]
        tr = lim_prm[i, 5]
        w_up = step_function(q, q_up, q_up + tr, 0.0, 1.0)
        w_low = step_function(q, q_low - tr, q_low, 1.0, 0.0)
        f = -(w_up * lim_prm[i, 3]) * (q - q_up) + (w_low * lim_prm[i, 2]) * (q_low - q) - lim_prm[i, 4] * (w_up + w_low) * qd
        qfrc[worldid, lim_adr[i, 1]] = qfrc[worldid, lim_adr[i, 1]] + f


@wp.kernel
def _advance(params: wp.array(dtype=vec21), family: wp.array(dtype=int), timestep: float, state: wp.array3d(dtype=float)):
    worldid, i = wp.tid()
    p = params[i]
    a = state[worldid, i, 0] + timestep * state[worldid, i, 5]
    if family[i] == 0:
        state[worldid, i, 0] = wp.clamp(a, p[P_AMIN], p[P_AMAX])
        if p[P_RIGID] > 0.5:
            state[worldid, i, 1] = 0.0
        else:
            state[worldid, i, 1] = wp.clamp(state[worldid, i, 1] + timestep * state[worldid, i, 6], p[P_LMIN], p[P_LMAX])
    else:
        state[worldid, i, 0] = wp.clamp(a, 0.0, 1.0)


@wp.kernel
def _equilibrate(params: wp.array(dtype=vec21), reset: wp.array(dtype=bool), state: wp.array3d(dtype=float)):
    worldid, i = wp.tid()
    # Bolt (smooth_muscle_contract._equilibrate): only worlds being reset are re-equilibrated.
    if not reset[worldid]:
        return
    state[worldid, i, 1] = equilibrium(state[worldid, i, 3], state[worldid, i, 4], state[worldid, i, 0], params[i])


class MuscleSystem:
    """Device-side muscle model + state for nworld worlds, mirroring the C plugin spec file."""

    def __init__(self, muscles, nworld, met_flags=15, aerobic=1.0, effort=1.0, device=None, limits=()):
        from .poly_utils import poly_exponents

        self.n = len(muscles)
        self.nworld = nworld
        self.met = (met_flags, aerobic, effort)
        f32 = np.float32
        dims, orders, nterms, exps, exp_adr, coefs, coef_adr = [], [], [], [], [], [], []
        qadr = np.zeros((self.n, MAXD), np.int32)
        dadr = np.zeros((self.n, MAXD), np.int32)
        gain = np.zeros((self.n, 10), f32)
        lr = np.zeros((self.n, 2), f32)
        acc0 = np.zeros(self.n, f32)
        dyn = np.zeros((self.n, 10), f32)
        ptt = np.full(self.n, -1, np.int32)
        for i, m in enumerate(muscles):
            assert m["path"] in ("poly", "poly+tendon", "tendon"), "Warp system supports polynomial/tendon paths"
            if m["path"] in ("poly+tendon", "tendon"):
                ptt[i] = m["tendon_id"]
            # "tendon" (pure native-tendon path, e.g. myo_sim's own muscles): no polynomial component at all.
            # _muscle_compute already unconditionally overwrites L/V from ten_length/ten_velocity whenever
            # pt_tendon[i]>=0, regardless of dim -- dim=0 here just makes the (unused) poly loop a no-op, and
            # _accumulate_tendon_qfrc (already used for Bolt's own poly+tendon muscle-0) applies the force via
            # the tendon Jacobian instead of a polynomial gradient. No kernel changes needed, only this wrapper.
            if m["path"] == "tendon":
                # poly_exponents(0, 0) still reports one (empty) term -- force nterm=0 explicitly instead, so
                # _muscle_compute's poly loop (`for k in range(nterm)`) never runs and no coefficient/exponent
                # array read happens for these muscles at all (L/V come entirely from ten_length/ten_velocity).
                dim_, order_, nterm_, e = 0, 0, 0, []
            else:
                dim_, order_ = m["dim"], m["order"]
                e = np.array(poly_exponents(dim_, order_), np.int32).reshape(-1).tolist()
                nterm_ = len(m["coef"])
            dims.append(dim_)
            orders.append(order_)
            nterms.append(nterm_)
            exp_adr.append(len(exps))
            exps.extend(e)
            coef_adr.append(len(coefs))
            if m["path"] != "tendon":
                coefs.extend(np.asarray(m["coef"], np.float64).tolist())
            if dim_:
                qadr[i, :dim_] = m["qposadr"]
                dadr[i, :dim_] = m["dofadr"]
            if m["family"] == 1:
                gain[i, :9] = m["gainprm"][:9]
                lr[i] = m["lengthrange"]
                acc0[i] = m["acc0"]
                dyn[i, :3] = m["dynprm"][:3]

        def A(x, dt):
            return wp.array(np.asarray(x), dtype=dt, device=device)

        self.params = A(np.stack([m["p"] for m in muscles]).astype(f32), vec21)
        self.family = A(np.array([m["family"] for m in muscles], np.int32), int)
        self.ctrl_id = A(np.array([m["ctrl_id"] for m in muscles], np.int32), int)
        self.dim = A(np.array(dims, np.int32), int)
        self.order = A(np.array(orders, np.int32), int)
        self.nterm = A(np.array(nterms, np.int32), int)
        self.exps = A(np.array(exps, np.int32), int)
        self.exp_adr = A(np.array(exp_adr, np.int32), int)
        self.coef = A(np.array(coefs, f32), float)
        self.coef_adr = A(np.array(coef_adr, np.int32), int)
        self.qposadr = A(qadr, int)
        self.dofadr = A(dadr, int)
        self.gainprm = A(gain, vec10)
        self.lengthrange = A(lr, wp.vec2)
        self.acc0 = A(acc0, float)
        self.dynprm = A(dyn, vec10)
        self.pt_tendon = A(ptt, int)
        self.pt_muscles_np = np.flatnonzero(ptt >= 0).astype(np.int32)
        self.pt_muscles = A(self.pt_muscles_np if len(self.pt_muscles_np) else np.zeros(1, np.int32), int)
        self.nlimit = len(limits)
        la = np.zeros((max(self.nlimit, 1), 2), np.int32)
        lp = np.zeros((max(self.nlimit, 1), 6), np.float32)
        for i, lim in enumerate(limits):
            la[i] = (lim["qposadr"], lim["dofadr"])
            lp[i] = [lim[k] for k in ("q_low", "q_up", "k_low", "k_up", "damping", "transition")]
        self.lim_adr = A(la, int)
        self.lim_prm = A(lp, float)
        self.state = wp.zeros((nworld, self.n, NSTATE), dtype=float, device=device)
        self.grad = wp.zeros((nworld, self.n, MAXD), dtype=float, device=device)

    def compute(self, d):
        wp.launch(
            _muscle_compute,
            dim=(self.nworld, self.n),
            inputs=[
                self.params,
                self.family,
                self.ctrl_id,
                self.dim,
                self.order,
                self.nterm,
                self.exps,
                self.exp_adr,
                self.coef,
                self.coef_adr,
                self.qposadr,
                self.dofadr,
                self.gainprm,
                self.lengthrange,
                self.acc0,
                self.dynprm,
                self.met[0],
                self.met[1],
                self.met[2],
                self.pt_tendon,
                d.qpos,
                d.qvel,
                d.ctrl,
                d.ten_length,
                d.ten_velocity,
            ],
            outputs=[self.state, self.grad],
        )

    def apply(self, d, m=None, zero=True):
        if zero:
            d.qfrc_applied.zero_()
        wp.launch(
            _accumulate_qfrc,
            dim=self.nworld,
            inputs=[self.n, self.dim, self.dofadr, self.state, self.grad],
            outputs=[d.qfrc_applied],
        )
        if len(self.pt_muscles_np):
            assert m is not None, "point-path muscles need the mujoco_warp model (tendon Jacobian)"
            wp.launch(
                _accumulate_tendon_qfrc,
                dim=(self.nworld, len(self.pt_muscles_np)),
                inputs=[self.pt_muscles, self.pt_tendon, m.ten_J_rownnz, m.ten_J_rowadr, m.ten_J_colind, d.ten_J, self.state],
                outputs=[d.qfrc_applied],
            )
        if self.nlimit:
            wp.launch(
                _limits,
                dim=self.nworld,
                inputs=[self.lim_adr, self.lim_prm, self.nlimit, d.qpos, d.qvel],
                outputs=[d.qfrc_applied],
            )

    def advance(self, timestep):
        wp.launch(_advance, dim=(self.nworld, self.n), inputs=[self.params, self.family, float(timestep)], outputs=[self.state])

    def equilibrate(self, d, reset=None):
        """Re-solve fiber length at force equilibrium for the worlds in ``reset`` (bool per world; None = all).
        Worlds not being reset keep their fiber state, as in Bolt."""
        self.compute(d)
        if reset is None:
            reset = np.ones(self.nworld, dtype=bool)
        if not isinstance(reset, wp.array):
            reset = wp.array(np.asarray(reset, dtype=bool).reshape(self.nworld), dtype=bool, device=self.state.device)
        wp.launch(_equilibrate, dim=(self.nworld, self.n), inputs=[self.params, reset], outputs=[self.state])

    def step(self, m, d, timestep):
        import mujoco_warp as mjw

        self.compute(d)
        self.apply(d)
        mjw.step(m, d)
        self.advance(timestep)


# --------------------------------------------------------------------------------------------- component batches
@wp.kernel
def _b_curves(
    x: wp.array(dtype=float),
    ws: wp.array(dtype=float),
    lmin: wp.array(dtype=float),
    e0: wp.array(dtype=float),
    strain: wp.array(dtype=float),
    out: wp.array2d(dtype=float),
):
    i = wp.tid()
    out[i, 0] = fl_active(x[i], ws[i])
    out[i, 1] = fv(x[i])
    out[i, 2] = fv_inv(x[i])
    out[i, 3] = fp(x[i], lmin[i], e0[i])
    out[i, 4] = ft(x[i], strain[i], False)
    out[i, 5] = ft_inv(x[i], strain[i])
    out[i, 6] = fv_deriv(x[i])


@wp.kernel
def _b_act(a: wp.array(dtype=float), u: wp.array(dtype=float), p: wp.array(dtype=vec21), out: wp.array(dtype=float)):
    i = wp.tid()
    out[i] = act_dot(a[i], u[i], p[i])


@wp.kernel
def _b_contraction(
    L: wp.array(dtype=float),
    V: wp.array(dtype=float),
    a: wp.array(dtype=float),
    ln: wp.array(dtype=float),
    p: wp.array(dtype=vec21),
    out: wp.array(dtype=vec12),
):
    i = wp.tid()
    out[i] = contraction(L[i], V[i], a[i], ln[i], p[i])


@wp.kernel
def _b_equilibrium(
    L: wp.array(dtype=float),
    V: wp.array(dtype=float),
    a: wp.array(dtype=float),
    p: wp.array(dtype=vec21),
    out: wp.array(dtype=float),
):
    i = wp.tid()
    out[i] = equilibrium(L[i], V[i], a[i], p[i])


@wp.kernel
def _b_metabolics(
    a: wp.array(dtype=float),
    u: wp.array(dtype=float),
    o: wp.array(dtype=vec12),
    p: wp.array(dtype=vec21),
    flags: int,
    aerobic: float,
    effort: float,
    out: wp.array(dtype=float),
):
    i = wp.tid()
    out[i] = metabolics(a[i], u[i], o[i], p[i], flags, aerobic, effort)


@wp.kernel
def _b_poly(
    dim: int,
    order: int,
    nterm: int,
    exps: wp.array(dtype=int),
    coef: wp.array(dtype=float),
    q: wp.array2d(dtype=float),
    out: wp.array2d(dtype=float),
):
    i = wp.tid()
    qq = vecd(0.0)
    for j in range(dim):
        qq[j] = q[i, j]
    r = poly_eval(dim, order, nterm, exps, 0, coef, 0, qq)
    for j in range(dim + 1):
        out[i, j] = r[j]


def _fa(x, dt=float):
    return wp.array(np.ascontiguousarray(np.asarray(x, np.float32)), dtype=dt)


OUT_KEYS = [
    "tendon_force",
    "norm_fiber_length",
    "norm_fiber_velocity",
    "fiber_velocity",
    "active_fiber_force",
    "fal",
    "fv",
    "fpe",
    "pennation",
    "ln_dot",
    "tendon_length",
    "clamped",
]


def b_curves(x, ws, lmin, e0, strain):
    out = wp.zeros((len(x), 7), dtype=float)
    wp.launch(_b_curves, dim=len(x), inputs=[_fa(x), _fa(ws), _fa(lmin), _fa(e0), _fa(strain), out])
    return dict(zip(["fl", "fv", "fv_inv", "fp", "ft", "ft_inv", "fv_deriv"], out.numpy().T))


def b_act(a, u, p):
    out = wp.zeros(len(a), dtype=float)
    wp.launch(_b_act, dim=len(a), inputs=[_fa(a), _fa(u), _fa(p, vec21), out])
    return out.numpy()


def b_contraction(L, V, a, ln, p):
    out = wp.zeros(len(L), dtype=vec12)
    wp.launch(_b_contraction, dim=len(L), inputs=[_fa(L), _fa(V), _fa(a), _fa(ln), _fa(p, vec21), out])
    o = out.numpy()
    d = dict(zip(OUT_KEYS, o.T))
    d["_raw"] = o
    return d


def b_equilibrium(L, V, a, p):
    out = wp.zeros(len(L), dtype=float)
    wp.launch(_b_equilibrium, dim=len(L), inputs=[_fa(L), _fa(V), _fa(a), _fa(p, vec21), out])
    return out.numpy()


def b_metabolics(a, u, cout, p, flags=15, aerobic=1.0, effort=1.0):
    out = wp.zeros(len(a), dtype=float)
    wp.launch(
        _b_metabolics,
        dim=len(a),
        inputs=[_fa(a), _fa(u), _fa(cout["_raw"], vec12), _fa(p, vec21), int(flags), float(aerobic), float(effort), out],
    )
    return out.numpy()


def b_poly(Q, coef, dim, order):
    from .poly_utils import poly_exponents

    exps = np.array(poly_exponents(dim, order), np.int32).reshape(-1)
    out = wp.zeros((Q.shape[0], MAXD + 1), dtype=float)
    wp.launch(
        _b_poly,
        dim=Q.shape[0],
        inputs=[int(dim), int(order), int(len(coef)), wp.array(exps, dtype=int), _fa(coef), _fa(Q), out],
    )
    o = out.numpy()
    return o[:, 0], o[:, 1 : dim + 1]
