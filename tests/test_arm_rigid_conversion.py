"""Arm muscle maps converted from MoBL-ARMS (myo_sim/build/rigid_conversion.py, region "arm").

Adopted maps match the MoBL-ARMS rigid-tendon fibre length over the full-range sample and stay inside the
force-length bounds. Motion-capture trajectories are not stored in the repository; the adopt decision is
recorded in ``build/data/mobl_arm_muscles.json``.
"""

import json

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
