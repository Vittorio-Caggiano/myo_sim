"""Build Bolt-parity MuscleSystem / ContactSystem assets for ``myofullbody_simple``.

Writes under ``myo_sim/models/fullbody_simple/``:
  - ``muscles.npy``  — DGF ``path="tendon"`` records (native MuJoCo tendon length + Jacobian)
  - ``contact.npz``  — Hunt–Crossley ``geom_flat`` / pairs (floor + foot/leg capsules)
  - ``map.json``     — qposadr / dofadr / limits for MuscleSystem joint-limit forces

Applies msk_envs EnvConfigSprinter-style F0×2 / vmax=12 boosts. ``lopt``/``lts`` come from
MuJoCo's own muscle mapping (``fiber_tendon_lengths``): optimal fiber length is MuJoCo's L0, not
the musculotendon length at optimum (that earlier bug made every fiber ~the whole MTU and
removed every tendon).

Repair the shipped ``muscles.npy`` in place (only lopt/lts change):
    python -m myo_sim.build.fullbody_simple_bolt_assets --repair-lengths

Formulation follows Bolt (Apache-2.0) and msk_envs (Apache-2.0); see NOTICE.

Run:
    python -m myo_sim.build.fullbody_simple_bolt_assets
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import mujoco
import numpy as np

OUT_DIR = Path(__file__).resolve().parents[1] / "models" / "fullbody_simple"

# Bolt / msk_envs EnvConfigSprinter contact defaults (ground-feet calibrated).
_STIFF = 1602213.464769315
_DISS = 0.0725
_FRICTION = (0.95, 0.3, 0.3)
_TRANS = 0.001

# Bolt myosprinter.xml joint viscous damping (N·m·s/rad). myofullbody_simple ships
# MyoArm/MyoLeg defaults (shoulder 1.05, elbow 0.75, legs/torso 0.5) which freeze
# arm swing and overdamp locomotor DOFs relative to Bolt.
_BOLT_DAMP_LEG = 0.1
_BOLT_DAMP_MTP = 0.4
_BOLT_DAMP_TORSO = 1.0
_BOLT_DAMP_SHOULDER = 0.1
_BOLT_DAMP_ELBOW = 0.01


def _bolt_damp_group(name: str) -> str | None:
    n = name.rsplit("/", 1)[-1].lower()
    if n.startswith("mtp_angle"):
        return "mtp"
    if any(n.startswith(p) for p in ("hip_", "knee_angle", "ankle_angle", "subtalar_angle")):
        return "leg"
    if n in ("flex_extension", "lat_bending", "axial_rotation"):
        return "torso"
    if ("elbow" in n) or n.startswith("pro_sup"):
        return "elbow"
    if any(s in n for s in ("shoulder", "elv_angle", "sternoclav", "acromioclav", "unrothum", "unrotscap")):
        return "shoulder"
    return None


def _bolt_damp_value(group: str) -> float:
    return {
        "leg": _BOLT_DAMP_LEG,
        "mtp": _BOLT_DAMP_MTP,
        "torso": _BOLT_DAMP_TORSO,
        "shoulder": _BOLT_DAMP_SHOULDER,
        "elbow": _BOLT_DAMP_ELBOW,
    }[group]


def apply_bolt_joint_damping(spec_or_model, *, arms: bool = True, legs: bool = True, torso: bool = True) -> int:
    """Set hinge damping to Bolt myosprinter homologs. Works on MjSpec or MjModel.

    Returns the number of joints changed. Scapular scaffold DOFs (no Bolt 1:1) use
    the shoulder value 0.1; pro_sup uses the elbow value 0.01.
    """
    want = set()
    if arms:
        want |= {"shoulder", "elbow"}
    if legs:
        want |= {"leg", "mtp"}
    if torso:
        want |= {"torso"}
    n_changed = 0
    if isinstance(spec_or_model, mujoco.MjSpec):
        for joint in spec_or_model.joints:
            if joint.type == mujoco.mjtJoint.mjJNT_FREE:
                continue
            g = _bolt_damp_group(joint.name or "")
            if g is None or g not in want:
                continue
            val = _bolt_damp_value(g)
            # MuJoCo >=3.8 polynomial damping: linear term in [0].
            cur = float(joint.damping[0]) if len(joint.damping) else 0.0
            if abs(cur - val) > 1e-12:
                joint.damping = [val, 0.0, 0.0]
                n_changed += 1
        return n_changed
    m = spec_or_model
    for j in range(m.njnt):
        if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE:
            continue
        name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j) or ""
        g = _bolt_damp_group(name)
        if g is None or g not in want:
            continue
        dof = int(m.jnt_dofadr[j])
        val = _bolt_damp_value(g)
        if abs(float(m.dof_damping[dof]) - val) > 1e-12:
            m.dof_damping[dof] = val
            n_changed += 1
    return n_changed


# Capsule/sphere/plane body geoms that participate in HC contact (exclude fingers).
# Foot/toe capsules are converted to spheres (Bolt sprinter style) — ContactSystem's
# plane–capsule path is unreliable for myo_sim's Z-up foot frames.
_CONTACT_NAME_SUBSTR = (
    "floor",
    "foot_col",
    "bofoot_col",
    "femur1_col",
    "femur2_col",
    "tibia1_col",
    "tibia2_col",
    "thorax_coll",
    "hat_cervical",
    "humerus_coll",
    "ulna_coll",
    "radius_coll",
)
_FOOT_SUBSTR = ("foot_col", "bofoot_col")


def _default_params(F0: float, lopt: float, lts: float = 0.2, alpha0: float = 0.0, vmax: float = 12.0):
    """DGF parameter vector (21 floats) matching Bolt / OpenSim DeGrooteFregly2016Muscle layout."""
    p = [0.0] * 21
    p[0] = F0
    p[1] = lopt
    p[2] = lts
    p[3] = alpha0
    p[4] = vmax
    p[5] = 0.1  # damping
    p[6] = 0.6  # e0
    p[7] = 0.049  # tendon_strain
    p[8] = 0.15  # width_scale
    p[9] = 0.5  # lmin
    p[10] = 1.6  # lmax
    # Activation dynamics and metabolic fields (same values as the shipped muscles.npy);
    # leaving them 0 gives tau_act=0 → act_dot=±inf → activation stuck at 0.
    p[11] = 0.01  # tau_act
    p[12] = 0.04  # tau_deact
    p[13] = 10.0  # act_smooth
    p[14] = 0.01  # amin
    p[15] = 1.0  # amax
    p[16] = 5.0e5  # specific tension
    p[17] = 1059.7  # density
    p[18] = 0.5  # slow-twitch fraction
    return p


def fiber_tendon_lengths(
    m: mujoco.MjModel, i: int, L_ref: float, ln_ok: tuple[float, float] = (0.6, 1.4)
) -> tuple[float, float, bool]:
    """DGF ``(lopt, lts)`` reproducing MuJoCo's muscle length map for actuator ``i``.

    MuJoCo: ``ln = range0 + (L - lengthrange0) / L0`` with ``L0 = Δlengthrange / Δrange``, so the
    optimal *fiber* length is L0 and a rigid tendon has ``lts = lengthrange0 - range0·L0``.
    (``lengthrange0 + (1 - range0)·L0`` is the musculotendon length at ln = 1, not lopt.)
    If the map is degenerate (lts ≤ 0, or ln at the reference pose outside ``ln_ok``), keep
    lopt = L0 and re-anchor lts so ln = 1 at ``L_ref``. Returns ``(lopt, lts, reanchored)``.
    """
    lr0, lr1 = float(m.actuator_lengthrange[i, 0]), float(m.actuator_lengthrange[i, 1])
    rng0, rng1 = float(m.actuator_gainprm[i, 0]), float(m.actuator_gainprm[i, 1])
    lopt = (lr1 - lr0) / (rng1 - rng0)
    lts = lr0 - rng0 * lopt
    ln = (L_ref - lts) / lopt
    if lts > 0 and ln_ok[0] <= ln <= ln_ok[1]:
        return lopt, lts, False
    lts = L_ref - lopt
    if lts <= 0.1 * L_ref:
        lts = 0.1 * L_ref
        lopt = L_ref - lts
    return lopt, lts, True


def _ref_lengths(m: mujoco.MjModel) -> np.ndarray:
    d = mujoco.MjData(m)
    d.qpos[:] = m.key_qpos[0] if m.nkey else m.qpos0
    mujoco.mj_forward(m, d)
    return d.ten_length.copy()


def build_muscles(m: mujoco.MjModel) -> list[dict]:
    """Build DGF tendon-path records; lopt/lts from ``fiber_tendon_lengths``."""
    ten_len = _ref_lengths(m)
    muscles: list[dict] = []
    n_reanchor = 0
    for i in range(m.nu):
        name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
        tid = int(m.actuator_trnid[i, 0])
        if name is None or tid < 0:
            continue
        if int(m.actuator_trntype[i]) != int(mujoco.mjtTrn.mjTRN_TENDON):
            continue
        F0 = float(m.actuator_gainprm[i, 2]) * 2.0  # EnvConfigSprinter F0×2
        lopt, lts, re = fiber_tendon_lengths(m, i, float(ten_len[tid]))
        n_reanchor += int(re)
        if not (math.isfinite(lopt) and lopt > 0 and math.isfinite(F0) and F0 > 0):
            raise RuntimeError(f"bad muscle params for {name}: lopt={lopt} F0={F0}")
        muscles.append(
            dict(
                p=_default_params(F0=F0, lopt=lopt, lts=lts, vmax=12.0),
                family=0,
                ctrl_id=i,
                path="tendon",
                tendon_id=tid,
                name=name,
            )
        )
    print(f"build_muscles: n={len(muscles)} reanchored={n_reanchor}")
    return muscles


def repair_lengths(m: mujoco.MjModel, path: Path) -> int:
    """Rewrite only lopt/lts (p[1], p[2]) of an existing muscles.npy with ``fiber_tendon_lengths``."""
    rows = list(np.load(path, allow_pickle=True))
    ten_len = _ref_lengths(m)
    n_reanchor = 0
    for r in rows:
        i = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, r["name"])
        if i < 0 or int(m.actuator_trnid[i, 0]) != int(r["tendon_id"]):
            raise RuntimeError(f"actuator/tendon mismatch for {r['name']}")
        lopt, lts, re = fiber_tendon_lengths(m, i, float(ten_len[r["tendon_id"]]))
        n_reanchor += int(re)
        r["p"][1] = lopt
        r["p"][2] = lts
    np.save(path, np.array(rows, dtype=object), allow_pickle=True)
    return n_reanchor


def build_map(m: mujoco.MjModel, muscles: list[dict]) -> dict:
    qposadr: dict[str, int] = {}
    dofadr: dict[str, int] = {}
    limits: list[dict] = []
    for j in range(m.njnt):
        name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j)
        if not name or int(m.jnt_type[j]) == int(mujoco.mjtJoint.mjJNT_FREE):
            continue
        qposadr[name] = int(m.jnt_qposadr[j])
        dofadr[name] = int(m.jnt_dofadr[j])
        lo, hi = float(m.jnt_range[j, 0]), float(m.jnt_range[j, 1])
        if hi > lo and abs(hi - lo) < 1e6:
            limits.append(
                dict(
                    coord=name,
                    q_low=lo,
                    q_up=hi,
                    k_low=500.0,
                    k_up=500.0,
                    damping=1.0,
                    transition=math.radians(1.0),
                )
            )
    return dict(
        qposadr=qposadr,
        dofadr=dofadr,
        muscles=[mu.get("name", f"m{i}") for i, mu in enumerate(muscles)],
        actuators=[mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, i) for i in range(m.nu)],
        limits=limits,
        springs=[],
    )


def _geom_row(m: mujoco.MjModel, gid: int, priority: int, name: str) -> list[float]:
    gtype = int(m.geom_type[gid])
    size = m.geom_size[gid].tolist()
    pos = m.geom_pos[gid].tolist()
    quat = m.geom_quat[gid].tolist()  # wxyz
    # ContactSystem (Bolt HC) treats plane normal as local +Y (OpenSim / Bolt Y-up).
    # myo_sim floors are MuJoCo Z-up planes (normal = local +Z). Rotate so local +Y → world +Z.
    # size[0]==0 marks an infinite plane (Bolt ground_collider convention).
    if gtype == int(mujoco.mjtGeom.mjGEOM_PLANE):
        s2 = math.sqrt(0.5)
        quat = [s2, s2, 0.0, 0.0]
        size = [0.0, 0.05, 0.0]  # infinite plane + small thickness hint
        aabb_c = [0.0, 0.0, 0.0]
        aabb_s = [0.0, 0.0, 0.0]
        rbound = 0.0
    elif gtype == int(mujoco.mjtGeom.mjGEOM_SPHERE) or any(s in name for s in _FOOT_SUBSTR):
        # Foot capsules → spheres at geom local origin (Bolt foot collider style).
        r = float(size[0])
        gtype = int(mujoco.mjtGeom.mjGEOM_SPHERE)
        size = [r, r, r]
        aabb_c = [0.0, 0.0, 0.0]
        aabb_s = [r, r, r]
        rbound = r
        quat = [1.0, 0.0, 0.0, 0.0]
    elif gtype == int(mujoco.mjtGeom.mjGEOM_CAPSULE):
        r, h = size[0], size[1]
        aabb_c = [0.0, 0.0, 0.0]
        aabb_s = [r, r + h, r]
        rbound = float(math.sqrt(r * r + h * h))
    else:
        raise ValueError(f"unsupported geom type {gtype}")
    fr = list(_FRICTION)
    # Use Bolt sprinter ground stiffness scale for feet/floor (not the raw 1.6e6 HC coeff).
    stiff = 13692.42123792177 if (gtype == 0 or any(s in name for s in _FOOT_SUBSTR)) else _STIFF
    return [
        float(m.geom_bodyid[gid]),
        float(gtype),
        *size,
        *pos,
        *quat,
        *fr,
        stiff,
        _DISS,
        _TRANS,
        float(priority),
        *aabb_c,
        *aabb_s,
        rbound,
    ]


def build_contact(m: mujoco.MjModel) -> dict:
    """Sprinter-style multi-sphere soles (floor pairs only).

    Foot frame on this model: local +Y → world +Z (up). Sole spheres use local_y so
    centres sit at world z≈r. No torso/arm capsule pairs — overlapping MJ capsules
    produce multi-kN HC self-collisions that destroy standing.
    """
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)

    rows: list[list[float]] = []
    names: list[str] = []
    # Floor
    floor_gid = None
    for gid in range(m.ngeom):
        if int(m.geom_type[gid]) == int(mujoco.mjtGeom.mjGEOM_PLANE):
            floor_gid = gid
            break
    if floor_gid is None:
        raise RuntimeError("no floor plane found")
    names.append(mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, floor_gid) or "floor")
    rows.append(_geom_row(m, floor_gid, 0, names[-1]))

    sole_r = 0.02
    foot_layouts = {
        "calcn_r": [(0.02, 0.0), (0.05, 0.02), (0.05, -0.02), (0.09, 0.0), (0.09, 0.025), (0.09, -0.025), (0.12, 0.0)],
        "calcn_l": [(0.02, 0.0), (0.05, -0.02), (0.05, 0.02), (0.09, 0.0), (0.09, -0.025), (0.09, 0.025), (0.12, 0.0)],
        "toes_r": [(0.01, 0.0), (0.03, 0.015), (0.03, -0.015), (0.05, 0.0)],
        "toes_l": [(0.01, 0.0), (0.03, -0.015), (0.03, 0.015), (0.05, 0.0)],
    }
    fr = list(_FRICTION)
    stiff = 13692.42123792177
    for bshort, xz in foot_layouts.items():
        bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, bshort)
        if bid < 0:
            # mjlab-prefixed
            for bi in range(m.nbody):
                n = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, bi) or ""
                if n.endswith("/" + bshort):
                    bid = bi
                    break
        if bid < 0:
            continue
        z0 = float(d.xpos[bid, 2])
        ly = sole_r - z0
        for i, (lx, lz) in enumerate(xz):
            r = sole_r
            rows.append(
                [
                    float(bid),
                    2.0,
                    r,
                    r,
                    r,
                    lx,
                    ly,
                    lz,
                    1.0,
                    0.0,
                    0.0,
                    0.0,
                    *fr,
                    stiff,
                    _DISS,
                    _TRANS,
                    1.0,
                    0.0,
                    0.0,
                    0.0,
                    r,
                    r,
                    r,
                    r,
                ]
            )
            names.append(f"{bshort}_{i + 1}")

    GF = np.ascontiguousarray(rows, np.float64)
    pairs = [[0, j] for j in range(1, len(rows))]
    naconmax = max(256, 8 * len(pairs))
    return dict(
        geom_flat=GF,
        pairs=np.asarray(pairs, np.int32),
        naconmax=int(naconmax),
        geom_names=np.array(names),
    )


def prepare_bolt_spec(spec: mujoco.MjSpec) -> mujoco.MjSpec:
    """Disable native MJ contacts; zero mjMUSCLE force; add 1e-3 armature (Bolt / WarpSprintEnv).

    Also aligns hinge ``damping`` to Bolt myosprinter homologs (legs 0.1, mtp 0.4,
    torso 1.0, shoulder scaffold 0.1, elbow/pro_sup 0.01). MyoArm/MyoLeg asset
    defaults otherwise freeze arm swing and overdamp locomotor DOFs (~5×).

    Tendons are kept so MuscleSystem path=\"tendon\" can read ten_length / ten_J. Actuators are
    rewritten as fixed-gain=0 generals so MuJoCo itself applies no muscle force (MuscleSystem
    owns qfrc via apply()).
    """
    spec.option.disableflags = int(mujoco.mjtDisableBit.mjDSBL_CONTACT)
    spec.option.timestep = 1e-4
    for joint in spec.joints:
        if joint.type != mujoco.mjtJoint.mjJNT_FREE:
            joint.armature = 1e-3
    apply_bolt_joint_damping(spec, arms=True, legs=True, torso=True)
    for act in spec.actuators:
        # Zero-force ctrl slot with filter dynamics so mj_model.na == nu (wd.act exists for
        # BoltMuscleAction / adapter reset writes). Tendon transmission kept for ten_length.
        act.dyntype = mujoco.mjtDyn.mjDYN_FILTER
        act.gaintype = mujoco.mjtGain.mjGAIN_FIXED
        act.biastype = mujoco.mjtBias.mjBIAS_NONE
        dyn = np.zeros(10)
        dyn[0] = 0.01
        act.dynprm = dyn
        act.gainprm = np.zeros(10)
        act.biasprm = np.zeros(10)
    return spec


def main() -> None:
    import sys

    from myo_sim import load_spec

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    # Compile the composed model (joint-stripped) with native mjMUSCLE params still present
    # so F0/lopt can be read from gainprm/lengthrange; runtime prepare_bolt_spec zeroes force.
    m = load_spec("myofullbody_simple").compile()

    if "--repair-lengths" in sys.argv:
        n = repair_lengths(m, OUT_DIR / "muscles.npy")
        meta_path = OUT_DIR / "meta.json"
        meta = json.loads(meta_path.read_text())
        meta.pop("root_cause_fix", None)
        meta["lts_calibration"] = (
            f"lopt = MuJoCo L0, lts = lengthrange0 - range0*L0 (rigid-tendon map); "
            f"{n} degenerate maps re-anchored to ln=1 at qpos0"
        )
        meta["fiber_length_fix"] = "2026-09-30: lopt was the MTU length at ln=1 (fibers ~whole MTU)"
        meta_path.write_text(json.dumps(meta, indent=2))
        print(f"repaired lopt/lts in {OUT_DIR / 'muscles.npy'} (reanchored={n})")
        return

    muscles = build_muscles(m)
    mapf = build_map(m, muscles)
    contact = build_contact(m)

    np.save(OUT_DIR / "muscles.npy", np.array(muscles, dtype=object), allow_pickle=True)
    (OUT_DIR / "map.json").write_text(json.dumps(mapf, indent=2))
    np.savez(
        OUT_DIR / "contact.npz",
        geom_flat=contact["geom_flat"],
        pairs=contact["pairs"],
        naconmax=contact["naconmax"],
        geom_names=contact["geom_names"],
    )
    meta = dict(
        nq=int(m.nq),
        nv=int(m.nv),
        nu=int(m.nu),
        njnt=int(m.njnt),
        n_muscles=len(muscles),
        n_contact_geoms=int(contact["geom_flat"].shape[0]),
        n_pairs=int(contact["pairs"].shape[0]),
        naconmax=int(contact["naconmax"]),
        n_limits=len(mapf["limits"]),
    )
    (OUT_DIR / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"wrote {OUT_DIR}: {meta}")


if __name__ == "__main__":
    main()
