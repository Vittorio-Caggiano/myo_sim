"""Smoke tests for the Bolt-like simplified fullbody scaffold."""

from __future__ import annotations

import mujoco

from myo_sim import load_model
from myo_sim.build.hand import RIGHT_HAND_ACTUATORS


def id_for(model, object_type, name: str) -> int:
    return mujoco.mj_name2id(model, object_type, name)


def actuator_names(model) -> set[str]:
    return {mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i) for i in range(model.nu)}


def test_myofullbody_simple_builds():
    model = load_model("myofullbody_simple")

    assert id_for(model, mujoco.mjtObj.mjOBJ_BODY, "sacrum") >= 0
    assert id_for(model, mujoco.mjtObj.mjOBJ_BODY, "pelvis") >= 0
    assert id_for(model, mujoco.mjtObj.mjOBJ_BODY, "humerus_r") >= 0
    assert id_for(model, mujoco.mjtObj.mjOBJ_JOINT, "root") >= 0
    assert id_for(model, mujoco.mjtObj.mjOBJ_JOINT, "flex_extension") >= 0
    assert id_for(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "ercspn_r") >= 0
    assert id_for(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "DELT1") >= 0
    assert id_for(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "DELT1_l") >= 0
    assert id_for(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "iliacus_r") >= 0


def test_myofullbody_simple_strips_distal_arm_muscles():
    model = load_model("myofullbody_simple")
    names = actuator_names(model)

    for base in RIGHT_HAND_ACTUATORS:
        assert base not in names, base
        assert f"{base}_l" not in names, f"{base}_l"

    assert "DELT1" in names
    assert "FDS5" not in names
    assert "EDC2" not in names
    assert "ECRL" not in names


def test_myofullbody_simple_keeps_abdomen_dof_set():
    model = load_model("myofullbody_simple")
    torso_joints = {
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
        for i in range(model.njnt)
        if mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
        in {"flex_extension", "lat_bending", "axial_rotation", "L4_L5_FE", "Abs_t1"}
    }
    assert torso_joints == {"flex_extension", "lat_bending", "axial_rotation"}


def test_myofullbody_simple_actuator_count_near_bolt():
    """Expect ~134 actuators: 6 trunk + 48 upper-arm + ~80 legs."""
    model = load_model("myofullbody_simple")
    assert 120 <= model.nu <= 150, model.nu
    assert model.nu < 300  # far below full myofullbody (416)


def test_myofullbody_simple_strips_distal_orphan_joints():
    """Wrist/hand joints with no remaining muscle must be deleted (welded fixed)."""
    model = load_model("myofullbody_simple")
    for name in (
        "deviation_r",
        "flexion_r",
        "cmc_flexion",
        "md3_flexion",
        "deviation_l",
        "flexion_l",
        "cmc_flexion_l",
    ):
        assert id_for(model, mujoco.mjtObj.mjOBJ_JOINT, name) < 0, name
    # nq drops by 44 vs the old passive-hand scaffold (114 → 70).
    assert model.nq == 70, model.nq
    assert model.njnt == 64, model.njnt
