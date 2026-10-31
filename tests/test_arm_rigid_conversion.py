"""Arm muscle maps converted from MoBL-ARMS (myo_sim/build/rigid_conversion.py, region "arm").

Adopted maps match the MoBL-ARMS rigid-tendon fibre length over the full-range sample, stay inside the force-length bounds,
and, for the shoulder muscles the Seth et al. (2019) thoracoscapular model tracks, keep their fibre lengths along that study's
measured shoulder motions close to the Seth model's (tests/data/seth2019_shoulder_motion.npz) and inside 0.5-1.5.
"""

import json
from pathlib import Path

import mujoco
import numpy as np
import pytest

import myo_sim
from myo_sim.build.rigid_conversion import (
    REGIONS,
    actuator_lengths,
    mujoco_fiber_length,
    rigid_equivalent_fiber_length,
    sample_postures,
)

SETH = Path(__file__).parent / "data" / "seth2019_shoulder_motion.npz"


@pytest.fixture(scope="module")
def arm():
    model, _ = myo_sim.load("myoarm")
    ref = json.loads(REGIONS["arm"]["data"].read_text())
    return model, ref


def _act(model, name: str) -> int:
    a = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
    assert a >= 0, name
    return a


def test_adopted_maps_match_mobl_and_bounds(arm):
    model, ref = arm
    postures = sample_postures(model, {k: tuple(v) for k, v in ref["posture_ranges_rad"].items()}, n=200, seed=7)
    acts = [_act(model, n) for n in ref["adopted"]]
    lengths = actuator_lengths(model, postures, acts)
    bad = {}
    for j, (name, a) in enumerate(zip(ref["adopted"], acts)):
        p = ref["muscles"][name]
        target = rigid_equivalent_fiber_length(
            lengths[:, j], p["optimal_fiber_length"], p["tendon_slack_length"], p["pennation_angle"]
        )
        rmse = float(np.sqrt(np.mean((mujoco_fiber_length(model, a, lengths[:, j]) - target) ** 2)))
        r0, r1, lmin, lmax = model.actuator_gainprm[a, [0, 1, 4, 5]]
        if not np.isfinite(rmse) or rmse > 0.15 or not (lmin < r0 < r1 < lmax):
            bad[name] = (round(rmse, 3), float(r0), float(r1), float(lmin), float(lmax))
    assert not bad, bad


def test_shoulder_fiber_length_along_measured_motion(arm):
    model, _ = arm
    z = np.load(SETH)
    cm = {"plane_elv": "elv_angle_r", "shoulder_elv": "shoulder_elv_r", "axial_rot": "shoulder_rot_r",
          "elbow_flexion": "elbow_flexion_r", "pro_sup": "pro_sup_r"}  # fmt: skip
    coords = [str(c) for c in z["coords"]]
    postures = np.stack(
        [
            sample_postures(model, {cm[c]: (np.radians(v), np.radians(v)) for c, v in zip(coords, row)}, 1, 0)[0]
            for row in z["angles_deg"]
        ]
    )
    names = [str(n) for n in z["muscles"]]
    lengths = actuator_lengths(model, postures, [_act(model, n) for n in names])
    for j, name in enumerate(names):
        ln = mujoco_fiber_length(model, _act(model, name), lengths[:, j])
        rmse = float(np.sqrt(np.mean((ln - z["seth_ln"][:, j]) ** 2)))
        assert rmse <= float(z["recorded_rmse"][j]) + 0.01, (name, rmse)
        assert np.mean((ln < 0.5) | (ln > 1.5)) <= 0.06, name
