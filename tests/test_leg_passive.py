"""Leg passive-force calibration (myo_sim/build/leg_passive.py) against Silder et al. (2007).

The fitted slack shifts and fpmax are in build/data/leg_passive.json ("fit"). The model's passive hip, knee and ankle moments
must match the analytic evaluation used by the fit, reproduce the recorded fit errors, and keep the active force-length
constraint at the angles of maximal voluntary strength (Anderson et al. 2007).
"""

import mujoco
import numpy as np
import pytest

import myo_sim
from myo_sim.build.leg_passive import (
    JOINTS,
    MIN_FL,
    T_MAX,
    LegMuscles,
    active_fl,
    geometry,
    load_passive_data,
    model_passive,
    silder_moments,
    silder_postures,
)


@pytest.fixture(scope="module")
def leg():
    model, data = myo_sim.load("myolegs")
    ref = load_passive_data()
    names = [n + "_r" for group in ref["groups"].values() for n in group]
    return model, data, ref, LegMuscles(model, names)


def _mj_passive(model, data, angles):
    return model_passive(model, angles)  # mj_forward forces times effective moment arms, all leg muscles


def test_fitted_errors_reproduce_and_improve(leg):
    model, data, ref, _ = leg
    ang = silder_postures(ref)
    target = silder_moments(ref["params"], ang[:, 0], ang[:, 1], ang[:, 2])
    ok = np.abs(target) < T_MAX
    err = np.where(ok, _mj_passive(model, data, ang) - target, np.nan)  # all leg muscles, as stored in the XML
    rms = np.sqrt(np.nanmean(err**2, axis=0))
    fit = ref["fit"]
    for i, j in enumerate(("hip", "knee", "ankle")):
        assert rms[i] <= fit["rms_Nm_after"][j] + 0.05, (j, rms[i])
        assert fit["rms_Nm_after"][j] < fit["rms_Nm_before"][j], j


def test_active_force_length_at_strength_postures(leg):
    model, _, ref, mus = leg
    for t in ref["strength_postures"].values():
        q = np.radians([t["posture_deg"].get(j, 0.0) for j in JOINTS])
        q[JOINTS.index(t["joint"])] = np.radians(t["peak_deg"])
        lengths, _ = geometry(model, q[None], mus.acts)
        fl = active_fl(mus.ln(lengths, np.zeros(len(mus.names)))[0], mus.lmin, mus.lmax)
        for n in t["muscles"]:
            i = mus.names.index(n + "_r")
            assert fl[i] >= MIN_FL - 1e-3, (n, t["joint"], fl[i])


def test_left_right_identical(leg):
    model, _, ref, _ = leg
    for group in ref["groups"].values():
        for n in group:
            r = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, n + "_r")
            left = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, n + "_l")
            np.testing.assert_allclose(model.actuator_gainprm[r], model.actuator_gainprm[left], err_msg=n)
            np.testing.assert_allclose(model.actuator_lengthrange[r], model.actuator_lengthrange[left], err_msg=n)
