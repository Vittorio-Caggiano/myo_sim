"""Rigid-tendon (MuJoCo native) muscle maps derived directly from an OpenSim reference muscle.

MuJoCo's muscle maps the actuator length ``L`` linearly to a normalised fibre length,
``ln = range0 + (L - lengthrange0) * (range1 - range0) / (lengthrange1 - lengthrange0)``, with a rigid tendon and no pennation.
For an OpenSim muscle with optimal fibre length ``l0``, tendon slack length ``lts`` and pennation ``alpha0`` (constant width), the
rigid-tendon fibre length is ``sqrt((L - lts)^2 + (l0 sin alpha0)^2) / l0``. ``fit_rigid_muscle_map`` fits MuJoCo's linear map to
it over sampled path lengths, so the converted muscle operates at the reference's fibre lengths without refitting ``l0`` to a
measured length range.

Reference values for the legs (Rajagopal et al. 2016) are packaged in ``data/rajagopal2016_leg_muscles.json``.

Usage (rewrites ``range`` and ``lengthrange`` of the adopted leg muscles, both sides, in ``myolegs_muscle.xml``)::

    uv run python -m myo_sim.build.rigid_conversion --write
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import mujoco
import numpy as np

DATA = Path(__file__).parent / "data" / "rajagopal2016_leg_muscles.json"
LEG_MUSCLE_XML = Path(__file__).parents[1] / "models" / "leg" / "assets" / "myolegs_muscle.xml"


def load_leg_reference() -> dict:
    return json.loads(DATA.read_text())


def rigid_equivalent_fiber_length(path_length: np.ndarray, l0: float, lts: float, alpha0: float) -> np.ndarray:
    """Normalised fibre length of a rigid-tendon muscle with constant-width pennation."""
    along = np.maximum(np.asarray(path_length) - lts, 0.0)
    return np.sqrt(along**2 + (l0 * np.sin(alpha0)) ** 2) / l0


def fit_rigid_muscle_map(
    path_length: np.ndarray, l0: float, lts: float, alpha0: float, lmin: float, lmax: float, margin: float = 0.02
) -> tuple[float, float, float, float]:
    """Least-squares MuJoCo map ``(range0, range1, lengthrange0, lengthrange1)`` for the sampled path lengths.

    The anchors are the path's sampled extremes, moved inward where the fitted map would leave ``(lmin, lmax)`` (MuJoCo
    extrapolates the same linear map beyond ``lengthrange``, where the force-length curve is zero).
    """
    path_length = np.asarray(path_length, float)
    slope, icpt = np.polyfit(path_length, rigid_equivalent_fiber_length(path_length, l0, lts, alpha0), 1)
    lr0, lr1 = float(path_length.min()), float(path_length.max())
    lr0 = max(lr0, (lmin + margin - icpt) / slope)
    lr1 = min(lr1, (lmax - margin - icpt) / slope)
    return float(icpt + slope * lr0), float(icpt + slope * lr1), lr0, lr1


def _joint_equalities(model: mujoco.MjModel) -> list[tuple[int, int, np.ndarray]]:
    return [
        (int(model.eq_obj1id[e]), int(model.eq_obj2id[e]), model.eq_data[e, :5].copy())
        for e in range(model.neq)
        if int(model.eq_type[e]) == int(mujoco.mjtEq.mjEQ_JOINT) and model.eq_active0[e] and model.eq_obj2id[e] >= 0
    ]


def sample_postures(model: mujoco.MjModel, ranges: dict[str, tuple[float, float]], n: int, seed: int) -> np.ndarray:
    """``n`` qpos vectors with the named joints uniform in ``ranges`` and joint-equality followers resolved."""
    rng = np.random.default_rng(seed)
    eqs = _joint_equalities(model)
    adr = {name: model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)] for name in ranges}
    out = np.tile(model.qpos0, (n, 1))
    for name, (lo, hi) in ranges.items():
        out[:, adr[name]] = rng.uniform(lo, hi, n)
    for _ in range(3):  # followers of followers
        for j1, j2, c in eqs:
            a1, a2 = model.jnt_qposadr[j1], model.jnt_qposadr[j2]
            x = out[:, a2] - model.qpos0[a2]
            out[:, a1] = model.qpos0[a1] + sum(c[k] * x**k for k in range(5))
    return out


def actuator_lengths(model: mujoco.MjModel, postures: np.ndarray, actuators: list[int]) -> np.ndarray:
    data = mujoco.MjData(model)
    out = np.zeros((len(postures), len(actuators)))
    for k, q in enumerate(postures):
        data.qpos[:] = q
        mujoco.mj_fwdPosition(model, data)
        out[k] = data.actuator_length[actuators]
    return out


def mujoco_fiber_length(model: mujoco.MjModel, actuator: int, length: np.ndarray) -> np.ndarray:
    """MuJoCo's normalised fibre length for actuator lengths ``length``."""
    r0, r1 = model.actuator_gainprm[actuator, :2]
    l0, l1 = model.actuator_lengthrange[actuator]
    return r0 + (np.asarray(length) - l0) * (r1 - r0) / (l1 - l0)


def leg_rigid_maps(model: mujoco.MjModel, n: int = 400, seed: int = 123) -> dict[str, tuple[float, float, float, float]]:
    """Fitted maps for the adopted right-leg muscles (apply to both sides; the leg is left/right symmetric)."""
    ref = load_leg_reference()
    postures = sample_postures(model, {k: tuple(v) for k, v in ref["posture_ranges_rad"].items()}, n, seed)
    names = ref["adopted"]
    acts = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name) for name in names]
    lengths = actuator_lengths(model, postures, acts)
    maps = {}
    for j, (name, a) in enumerate(zip(names, acts)):
        p = ref["muscles"][name]
        lmin, lmax = (float(v) for v in model.actuator_gainprm[a, 4:6])
        maps[name] = fit_rigid_muscle_map(
            lengths[:, j], p["optimal_fiber_length"], p["tendon_slack_length"], p["pennation_angle"], lmin, lmax
        )
    return maps


def write_leg_maps(maps: dict[str, tuple[float, float, float, float]], xml_path: Path = LEG_MUSCLE_XML) -> int:
    """Rewrite range (gainprm/biasprm[0:2]) and lengthrange of the right muscles and their ``_l`` mirrors in place."""
    text = xml_path.read_text()
    n = 0
    for right, (r0, r1, l0, l1) in maps.items():
        for name in (right, right[:-2] + "_l"):
            pat = re.compile(r'(<general[^>]*name="' + re.escape(name) + r'"[^>]*/>)')
            match = pat.search(text)
            if match is None:
                raise KeyError(name)
            line = match.group(1)
            for attr in ("gainprm", "biasprm"):
                line = re.sub(attr + r'="([^ "]+) ([^ "]+) ', lambda _m: f'{attr}="{r0:.6f} {r1:.6f} ', line, count=1)
            line = re.sub(r'lengthrange="[^"]*"', f'lengthrange="{l0:.6f} {l1:.6f}"', line, count=1)
            text = text[: match.start(1)] + line + text[match.end(1) :]
            n += 1
    xml_path.write_text(text)
    return n


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--write", action="store_true", help="rewrite myolegs_muscle.xml (default: print the maps)")
    args = ap.parse_args()
    import myo_sim

    model, _ = myo_sim.load("myolegs")
    maps = leg_rigid_maps(model)
    for name, (r0, r1, l0, l1) in maps.items():
        print(f"{name:14s} range {r0:.6f} {r1:.6f}  lengthrange {l0:.6f} {l1:.6f}")
    if args.write:
        print("rewrote", write_leg_maps(maps), "actuators in", LEG_MUSCLE_XML)


if __name__ == "__main__":
    main()
