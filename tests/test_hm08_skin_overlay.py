"""HM08 full-body ``.skn`` attaches procedurally to whole-body and segment models."""

from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np
import pytest

from myo_sim.skin import (
    HM08_SKN,
    SEGMENTS,
    attach_skin,
    filter_skin,
    load_model_with_skin,
    read_skn,
    write_skn,
)


@pytest.fixture(scope="module")
def hm08_skin():
    assert HM08_SKN.is_file(), f"Missing packaged SKN: {HM08_SKN}"
    return read_skn(HM08_SKN)


def test_hm08_skn_header(hm08_skin):
    assert hm08_skin.vertices.shape[0] > 10_000
    assert hm08_skin.faces.shape[0] > 20_000
    assert len(hm08_skin.bones) >= 50
    bodies = {b.body for b in hm08_skin.bones}
    for required in ("pelvis", "torso", "humerus_r", "femur_l", "head"):
        assert required in bodies


def test_skn_roundtrip(tmp_path: Path, hm08_skin):
    out = tmp_path / "roundtrip.skn"
    write_skn(out, hm08_skin)
    again = read_skn(out)
    assert again.vertices.shape == hm08_skin.vertices.shape
    assert again.faces.shape == hm08_skin.faces.shape
    assert len(again.bones) == len(hm08_skin.bones)
    assert np.allclose(again.vertices, hm08_skin.vertices)
    assert np.array_equal(again.faces, hm08_skin.faces)


def test_attach_fullbody():
    model, data = load_model_with_skin("myofullbody")
    assert model.nskin == 1
    assert model.nskinvert == read_skn(HM08_SKN).vertices.shape[0]
    mujoco.mj_step(model, data)
    assert np.all(np.isfinite(data.xpos))


def test_attach_fullbody_simple():
    model, data = load_model_with_skin("myofullbody_simple")
    assert model.nskin == 1
    mujoco.mj_forward(model, data)
    assert np.all(np.isfinite(data.xpos))


@pytest.mark.parametrize(
    ("model_name", "segment"),
    [
        ("myoarm_r", "arm_r"),
        ("myolegs", "legs"),
        ("myotorso", "torso"),
        ("myotorso_arms", "arm_l"),
    ],
)
def test_attach_segment(model_name: str, segment: str):
    model, data = load_model_with_skin(model_name, segment=segment, skin_name=f"{segment}_skin")
    assert model.nskin == 1
    assert model.nskinvert > 500
    bone_names = {
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, int(bid)) for bid in model.skin_bonebodyid[: model.nskinbone]
    }
    # Every remaining bind body must be in the requested segment and present.
    assert bone_names <= SEGMENTS[segment]
    mujoco.mj_forward(model, data)
    assert np.all(np.isfinite(data.xpos))


def test_filter_skin_arm_r(hm08_skin):
    filtered = filter_skin(hm08_skin, SEGMENTS["arm_r"])
    assert {b.body for b in filtered.bones} <= SEGMENTS["arm_r"]
    assert filtered.vertices.shape[0] < hm08_skin.vertices.shape[0]
    assert filtered.faces.shape[0] < hm08_skin.faces.shape[0]


def test_attach_via_filtered_file(tmp_path: Path):
    from myo_sim import load_spec

    spec = load_spec("myolegs")
    out = tmp_path / "legs.skn"
    attach_skin(spec, segment="legs", filtered_skn_path=out, name="legs_skin")
    assert out.is_file()
    model = spec.compile()
    assert model.nskin == 1
    assert read_skn(out).vertices.shape[0] == model.nskinvert
