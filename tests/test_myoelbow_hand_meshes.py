"""Legacy myoelbow must use the ulna-frame arm_r_* STLs, not myoarm local-frame bones."""

from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np
import pytest

import myo_sim

HAND_GEOMS = (
    "arm_r_lunate",
    "arm_r_1mc",
    "arm_r_2proxph",
    "arm_r_5distph",
    "arm_r_thumbdist",
)

ELBOW_MODELS = (
    "legacy/elbow/myoelbow_1dof6muscles.xml",
    "legacy/elbow/myoelbow_2dof6muscles.xml",
    "legacy/elbow/myoelbow_1dof6muscles_1dofexo.xml",
)


def test_arm_r_hand_stls_are_not_myoarm_local_frames() -> None:
    meshes = myo_sim.MODELS_DIR / "meshes"
    assert (meshes / "arm_r_lunate.stl").read_bytes() != (meshes / "lunate.stl").read_bytes()


@pytest.mark.parametrize("rel", ELBOW_MODELS)
def test_hand_geoms_sit_distal_to_the_elbow(rel: str) -> None:
    model = mujoco.MjModel.from_xml_path(str(myo_sim.MODELS_DIR / rel))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    ulna = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "r_ulna_radius_hand")
    assert ulna >= 0
    elbow = data.xpos[ulna]

    for name in HAND_GEOMS:
        gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
        assert gid >= 0, name
        dist = float(np.linalg.norm(data.geom_xpos[gid] - elbow))
        assert dist > 0.15, f"{name} is only {dist:.3f} m from the elbow (still using local-frame STLs)"


def test_myosuite_scene_meshdir_override_still_finds_hand_meshes(tmp_path: Path) -> None:
    """MyoSuite wraps elbow assets with the scene compiler meshdir=MODELS_DIR."""
    wrapper = tmp_path / "myoelbow_myosuite_wrap.xml"
    wrapper.write_text(
        f"""<mujoco>
  <include file="{myo_sim.MODELS_DIR / "legacy/elbow/assets/myoelbow_assets.xml"}"/>
  <include file="{myo_sim.MODELS_DIR / "legacy/elbow/assets/myoelbow_1dof6muscles_body.xml"}"/>
  <include file="{myo_sim.MODELS_DIR / "scene/myosuite_scene.xml"}"/>
  <compiler meshdir="{myo_sim.MODELS_DIR}" texturedir="{myo_sim.MODELS_DIR}"/>
</mujoco>
"""
    )
    model = mujoco.MjModel.from_xml_path(str(wrapper))
    gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "arm_r_lunate")
    assert gid >= 0
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    ulna = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "r_ulna_radius_hand")
    dist = float(np.linalg.norm(data.geom_xpos[gid] - data.xpos[ulna]))
    assert dist > 0.15
