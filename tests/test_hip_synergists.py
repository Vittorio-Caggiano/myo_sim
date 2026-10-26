import mujoco
import numpy as np
import pytest

from myo_sim import load_spec
from myo_sim.build.synergists import HIP_SYNERGISTS

NEW = [f"{m.name}_{s}" for s in ("r", "l") for m in HIP_SYNERGISTS]
HIP = ("hip_flexion", "hip_adduction", "hip_rotation")


def actuator_names(model):
    return [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i) for i in range(model.nu)]


@pytest.fixture(scope="module")
def legs():
    return load_spec("myolegs", hip_synergists=True).compile()


@pytest.mark.parametrize("model_name", ("myolegs", "myofullbody"))
def test_default_models_do_not_contain_the_synergists(model_name):
    assert not set(NEW) & set(actuator_names(load_spec(model_name).compile()))


@pytest.mark.parametrize("model_name", ("myolegs", "myofullbody"))
def test_opt_in_appends_six_actuators_and_keeps_existing_order(model_name):
    base = actuator_names(load_spec(model_name).compile())
    extended = actuator_names(load_spec(model_name, hip_synergists=True).compile())
    assert extended[: len(base)] == base
    assert sorted(extended[len(base) :]) == sorted(NEW)


def tendon_length(model, data, name, q):
    data.qpos[:] = model.qpos0
    for joint, value in q.items():
        data.qpos[model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint)]] = value
    mujoco.mj_kinematics(model, data)
    mujoco.mj_tendon(model, data)
    return data.ten_length[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_TENDON, f"{name}_tendon")]


def moment_arm(model, data, muscle, joint, q0=None, h=1e-4):
    q0 = dict(q0 or {})
    plus, minus = dict(q0), dict(q0)
    plus[joint] = q0.get(joint, 0.0) + h
    minus[joint] = q0.get(joint, 0.0) - h
    return -(tendon_length(model, data, muscle, plus) - tendon_length(model, data, muscle, minus)) / (2 * h)


# gait2392 values at neutral (flexion, adduction, rotation, mm); tolerance covers the ~11 mm frame-transfer error
GAIT2392 = {"pect": (23.1, 33.0, 2.2), "quad_fem": (4.3, 42.0, -40.2), "gem": (1.3, 11.6, -32.0)}


@pytest.mark.parametrize("muscle", GAIT2392)
def test_hip_moment_arms_agree_with_gait2392(legs, muscle):
    data = mujoco.MjData(legs)
    arms = [1000 * moment_arm(legs, data, f"{muscle}_r", f"{j}_r") for j in HIP]
    np.testing.assert_allclose(arms, GAIT2392[muscle], atol=10.0)


@pytest.mark.parametrize("muscle", GAIT2392)
def test_left_right_lengths_are_mirrored(legs, muscle):
    data = mujoco.MjData(legs)
    q = {"flexion": 0.4, "adduction": 0.2, "rotation": -0.3}
    right = tendon_length(legs, data, f"{muscle}_r", {f"hip_{k}_r": v for k, v in q.items()})
    left = tendon_length(legs, data, f"{muscle}_l", {f"hip_{k}_l": v for k, v in q.items()})
    assert right == pytest.approx(left, abs=1e-6)


def test_fibre_range_is_consistent_with_lengthrange_and_inside_curve_bounds(legs):
    for m in HIP_SYNERGISTS:
        assert 0.05 < m.fibre_range[0] < m.fibre_range[1] < 3.5
        assert m.lengthrange[0] < m.lengthrange[1]
        for side in ("r", "l"):
            act = mujoco.mj_name2id(legs, mujoco.mjtObj.mjOBJ_ACTUATOR, f"{m.name}_{side}")
            np.testing.assert_allclose(legs.actuator_lengthrange[act], m.lengthrange, atol=1e-6)


def test_random_activation_rollout_is_stable():
    model = load_spec("myofullbody", hip_synergists=True).compile()
    data = mujoco.MjData(model)
    rng = np.random.default_rng(0)
    for _ in range(300):
        data.ctrl[:] = rng.random(model.nu)
        mujoco.mj_step(model, data)
    assert np.isfinite(data.qpos).all()
