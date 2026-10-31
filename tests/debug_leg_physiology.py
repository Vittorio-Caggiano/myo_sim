"""Compare the myolegs muscles (MuJoCo native, rigid tendon) with human experimental data. Standalone analysis, not pytest.

References:
- Anderson DE, Madigan ML, Nussbaum MA (2007) Maximum voluntary joint torque as a function of joint angle and angular
  velocity: model development and application to the lower limb. J Biomech 40:3105-3113. doi:10.1016/j.jbiomech.2007.03.022
  (young males, Table 3; isometric torque C1*BW*H*cos(C2*(theta - C3)), flexion/dorsiflexion positive; test postures below).
- Ward SR, Eng CM, Smallwood LH, Lieber RL (2009) Are current measurements of lower extremity muscle architecture accurate?
  Clin Orthop Relat Res 467:1074-1082. doi:10.1007/s11999-008-0594-8 (Table 3, fibre length normalised to 2.7 um sarcomeres).
  Passive knee moment with the hip at 70 deg and the knee extended: 6.25 Nm (young males, B2 = -6.25, k2 = -4.521).
- Narici MV, Binzoni T, Hiltbrand E, Fasel J, Terrier F, Cerretelli P (1996) In vivo human gastrocnemius architecture with
  changing joint angle at rest and during graded isometric contraction. J Physiol 496:287-297.
  doi:10.1113/jphysiol.1996.sp021685 (medial gastrocnemius fascicles at rest, knee extended: 57.0 mm at 0 deg to 34.0 mm at
  60 deg plantarflexion).

Active torque = all agonists (moment-arm sign) at full activation minus the zero-activation torque, at zero velocity; MVC is
voluntary, so model/human ratios slightly above 1 are expected. Output: tests/output/leg_physiology/summary.json.

    cd tests && uv run python debug_leg_physiology.py
"""

import json
from pathlib import Path

import mujoco
import numpy as np

import myo_sim
from myo_sim.build.rigid_conversion import effective_moment_arms, sample_postures

HEIGHT = 1.748  # m, mean height of Anderson et al.'s young males; body weight from the model's mass
TESTS = {  # joint, tested range (deg), fixed joints (rad): Anderson et al. 2007 dynamometer postures
    "ankle": ("ankle_angle_r", (-40, 25), {"hip_flexion_r": np.radians(80), "knee_angle_r": np.radians(50)}),
    "knee": ("knee_angle_r", (0, 110), {"hip_flexion_r": np.radians(70)}),
    "hip": ("hip_flexion_r", (-20, 100), {"knee_angle_r": np.radians(10)}),
}
ANDERSON = {  # (joint, +1 flexion / -1 extension): C1 (BW*H), C2, C3 (rad)
    ("hip", -1): (0.161, 0.958, 0.932),
    ("hip", 1): (0.113, 0.738, -0.214),
    ("knee", -1): (0.163, 1.258, 1.133),
    ("knee", 1): (0.087, 0.869, 0.522),
    ("ankle", -1): (0.095, 1.391, 0.408),
    ("ankle", 1): (0.033, 1.510, -0.187),
}
ACTIONS = {
    ("hip", -1): "hip extension",
    ("hip", 1): "hip flexion",
    ("knee", -1): "knee extension",
    ("knee", 1): "knee flexion",
    ("ankle", -1): "plantarflexion",
    ("ankle", 1): "dorsiflexion",
}
WARD_CM = {  # optimal fibre length, mean and SD (cm); multi-part muscles compare each part with the whole muscle
    "psoas": (11.69, 1.66), "iliacus": (10.66, 1.86), "glmax": (15.69, 2.57), "glmed": (7.33, 1.57), "sart": (40.30, 4.63),
    "recfem": (7.59, 1.28), "vaslat": (9.94, 1.76), "vasint": (9.93, 2.03), "vasmed": (9.68, 2.30), "grac": (22.78, 4.38),
    "addlong": (10.82, 2.02), "addbrev": (10.31, 1.42), "addmag": (14.44, 2.74), "bflh": (9.76, 2.62), "bfsh": (11.03, 2.06),
    "semiten": (19.30, 4.12), "semimem": (6.90, 1.83), "tibant": (6.83, 0.79), "ehl": (7.48, 1.13), "edl": (6.93, 1.14),
    "perlong": (5.08, 0.63), "perbrev": (4.54, 0.65), "gasmed": (5.10, 0.98), "gaslat": (5.88, 0.95), "soleus": (4.40, 0.99),
    "fhl": (5.27, 1.29), "fdl": (4.46, 1.06), "tibpost": (3.78, 0.49),
}  # fmt: skip


def pose(model: mujoco.MjModel, joints: dict[str, float]) -> np.ndarray:
    return sample_postures(model, {k: (v, v) for k, v in joints.items()}, n=1, seed=0)[0]


def map_l0(model: mujoco.MjModel, a: int) -> float:
    """MuJoCo's optimal fibre length: lengthrange span over range span."""
    r0, r1 = model.actuator_gainprm[a, :2]
    l0, l1 = model.actuator_lengthrange[a]
    return float((l1 - l0) / (r1 - r0))


def torque_curves(model: mujoco.MjModel, data: mujoco.MjData) -> dict:
    """Maximal isometric active torque about the tested coordinate: actuator forces times effective moment arms (dL/dtheta
    with joint-equality followers such as the knee translations resolved), all agonists at full activation minus zero."""
    muscle = [a for a in range(model.nu) if int(model.actuator_gaintype[a]) == int(mujoco.mjtGain.mjGAIN_MUSCLE)]
    out = {}
    for test, (joint, (lo, hi), fixed) in TESTS.items():
        angles = np.arange(lo, hi + 1e-9, 5.0)
        curves = {1: [], -1: []}
        for ang in angles:
            joints = {**fixed, joint: np.radians(ang)}
            _, arm = effective_moment_arms(model, joints, [joint], muscle)
            data.qpos[:] = pose(model, joints)
            data.qvel[:] = 0
            data.act[:] = 0
            mujoco.mj_forward(model, data)
            base = data.actuator_force[muscle] @ arm[:, 0]
            for sign in (1, -1):
                agonists = [a for a, r in zip(muscle, arm[:, 0]) if sign * r < -1e-3]  # dL/dtheta < 0 shortens: +torque
                data.act[:] = 0
                data.act[model.actuator_actadr[agonists]] = 1.0
                mujoco.mj_forward(model, data)
                curves[sign].append(sign * float(data.actuator_force[muscle] @ arm[:, 0] - base))
        out[test] = (angles, curves)
    return out


def main() -> None:
    model, data = myo_sim.load("myolegs")
    bwh = float(np.sum(model.body_mass)) * 9.81 * HEIGHT
    summary = {"strength": {}, "fiber_length_vs_ward": {}, "gasmed_fascicle_rest_mm": {}}
    muscles = [a for a in range(model.nu) if int(model.actuator_gaintype[a]) == int(mujoco.mjtGain.mjGAIN_MUSCLE)]
    joints = {"hip_flexion_r": np.radians(70), "knee_angle_r": 0.0}
    _, arm = effective_moment_arms(model, joints, ["knee_angle_r"], muscles)
    data.qpos[:] = pose(model, joints)
    data.act[:] = 0
    mujoco.mj_forward(model, data)
    passive = float(data.actuator_force[muscles] @ arm[:, 0])
    summary["passive_knee_moment_hip70_knee0_Nm"] = {"model": round(passive, 1), "human": 6.25}
    curves = torque_curves(model, data)
    for (test, sign), (c1, c2, c3) in ANDERSON.items():
        angles, cur = curves[test]
        theta = np.radians(angles)
        human = c1 * bwh * np.cos(c2 * (theta - c3))
        valid = np.abs(c2 * (theta - c3)) < np.pi / 2
        y = np.array(cur[sign])
        shape = np.sqrt(np.mean((y[valid] / y.max() - human[valid] / human.max()) ** 2))
        summary["strength"][ACTIONS[(test, sign)]] = {
            "peak_Nm": round(float(y.max()), 1),
            "peak_deg": float(angles[y.argmax()]),
            "human_peak_Nm": round(c1 * bwh, 1),
            "human_peak_deg": round(float(np.degrees(c3)), 1),
            "ratio": round(float(y.max() / (c1 * bwh)), 2),
            "shape_rmse": round(float(shape), 3),
        }
    within, logs = 0, []
    for a in range(model.nu):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, a) or ""
        if not name.endswith("_r"):
            continue
        key = next((k for k in sorted(WARD_CM, key=len, reverse=True) if name.startswith(k)), None)
        if key is None:
            continue
        mean, sd = WARD_CM[key]
        l0 = 100 * map_l0(model, a)
        within += abs(l0 - mean) <= sd
        logs.append(np.log(l0 / mean))
        summary["fiber_length_vs_ward"][name] = {"model_cm": round(l0, 2), "ward_cm": mean, "ward_sd_cm": sd}
    summary["fiber_length_vs_ward_summary"] = {
        "muscles": len(logs),
        "within_1sd": int(within),
        "mean_abs_error_pct": round(100 * (np.exp(np.mean(np.abs(logs))) - 1), 1),
    }
    gm = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "gasmed_r")
    for ang in (0, -20, -40):
        data.qpos[:] = pose(model, {"hip_flexion_r": 0.0, "knee_angle_r": 0.0, "ankle_angle_r": np.radians(ang)})
        mujoco.mj_fwdPosition(model, data)
        r0, r1 = model.actuator_gainprm[gm, :2]
        ln = r0 + (data.actuator_length[gm] - model.actuator_lengthrange[gm, 0]) / map_l0(model, gm)
        summary["gasmed_fascicle_rest_mm"][ang] = {
            "model": round(1000 * float(ln) * map_l0(model, gm), 1),
            "narici_1996": round(57.0 + (34.0 - 57.0) * (-ang) / 60.0, 1),  # linear between the reported end points
        }
    out = Path(__file__).parent / "output" / "leg_physiology"
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    for action, s in summary["strength"].items():
        print(f"{action:15s} model {s['peak_Nm']:6.0f} Nm @ {s['peak_deg']:4.0f} | human {s['human_peak_Nm']:5.0f} @ "
              f"{s['human_peak_deg']:5.1f} | x{s['ratio']} shape {s['shape_rmse']}")  # fmt: skip
    print("passive knee moment, hip 70 deg, knee 0 (Nm):", summary["passive_knee_moment_hip70_knee0_Nm"])
    print("fibre length vs Ward 2009:", summary["fiber_length_vs_ward_summary"])
    print("gasmed fascicle at rest (mm):", summary["gasmed_fascicle_rest_mm"])


if __name__ == "__main__":
    main()
