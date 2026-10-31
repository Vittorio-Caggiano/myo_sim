"""Leg muscle maps reproduce the reference (Rajagopal et al. 2016) rigid-tendon fibre lengths.

For the muscles converted with ``myo_sim.build.rigid_conversion`` (listed as ``adopted`` in
``myo_sim/build/data/rajagopal2016_leg_muscles.json``), MuJoCo's normalised fibre length over a full-range posture sample must
match the reference's rigid-tendon fibre length, left and right maps must be identical, and the maps must stay inside the
force-length bounds. The semimembranosus check guards the regression where an oversized fit put it at ln ~1.9 standing
(Rajagopal: ~0.95).
"""

import mujoco
import numpy as np
import pytest

import myo_sim
from myo_sim.build.rigid_conversion import (
    actuator_lengths,
    load_leg_reference,
    mujoco_fiber_length,
    rigid_equivalent_fiber_length,
    sample_postures,
)

# RMSE of MuJoCo's linear map against the (slightly nonlinear, pennated) rigid-tendon reference over the full-range sample.
# semimem is the worst fit (~0.12 on this sample, ~0.10 against OpenSim's Millard fibre length): its path spans ~2.3 optimal
# fibre lengths and the linear map cannot follow the reference's floor at slack-tendon lengths. The others fit to < 0.06.
MAX_LN_RMSE = 0.15


@pytest.fixture(scope="module")
def leg():
    model, _ = myo_sim.load("myolegs")
    ref = load_leg_reference()
    postures = sample_postures(model, {k: tuple(v) for k, v in ref["posture_ranges_rad"].items()}, n=200, seed=7)
    return model, ref, postures


def _act(model, name: str) -> int:
    a = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
    assert a >= 0, name
    return a


def test_adopted_maps_match_reference_fiber_length(leg):
    model, ref, postures = leg
    names = ref["adopted"]
    acts = [_act(model, n) for n in names]
    lengths = actuator_lengths(model, postures, acts)
    failures = {}
    for j, (name, a) in enumerate(zip(names, acts)):
        p = ref["muscles"][name]
        target = rigid_equivalent_fiber_length(
            lengths[:, j], p["optimal_fiber_length"], p["tendon_slack_length"], p["pennation_angle"]
        )
        rmse = float(np.sqrt(np.mean((mujoco_fiber_length(model, a, lengths[:, j]) - target) ** 2)))
        if rmse > MAX_LN_RMSE:
            failures[name] = round(rmse, 3)
    assert not failures, failures


def test_adopted_maps_are_left_right_identical_and_inside_force_length_bounds(leg):
    model, ref, _ = leg
    for name in ref["adopted"]:
        r, left = _act(model, name), _act(model, name[:-2] + "_l")
        np.testing.assert_allclose(model.actuator_gainprm[r, :2], model.actuator_gainprm[left, :2], err_msg=name)
        np.testing.assert_allclose(model.actuator_lengthrange[r], model.actuator_lengthrange[left], err_msg=name)
        np.testing.assert_allclose(model.actuator_gainprm[r, :2], model.actuator_biasprm[r, :2], err_msg=name)
        r0, r1, lmin, lmax = model.actuator_gainprm[r, [0, 1, 4, 5]]
        assert lmin < r0 < 1.0 < r1 < lmax, (name, r0, r1, lmin, lmax)


def test_semimem_standing_fiber_length_matches_reference(leg):
    model, ref, _ = leg
    data = mujoco.MjData(model)
    mujoco.mj_fwdPosition(model, data)  # qpos0: hip, knee and ankle at 0
    a = _act(model, "semimem_r")
    ln = float(mujoco_fiber_length(model, a, data.actuator_length[a]))
    assert 0.7 < ln < 1.3, ln  # Rajagopal2016 (OpenSim, passive): 0.95
