"""Tests for human-atlas full-body skin overlay (Phase 1)."""

from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np
import pytest
import yaml

ATLAS_DIR = Path(__file__).resolve().parents[1] / "myo_sim" / "models" / "visual" / "atlas"
MAPPING = ATLAS_DIR / "mapping.yaml"
SKIN_XML = ATLAS_DIR / "myofullbody_atlas_skin.xml"
SCRIPTS = Path(__file__).resolve().parents[1] / "scripts" / "atlas"


def test_attribution_present():
    assert (ATLAS_DIR / "ATTRIBUTION.md").exists()
    text = (ATLAS_DIR / "ATTRIBUTION.md").read_text()
    assert "BodyParts3D" in text
    assert "CC BY 4.0" in text or "CC Attribution 4.0" in text


def test_genital_denylist_in_mapping():
    mapping = yaml.safe_load(MAPPING.read_text())
    deny = " ".join(mapping["genital_denylist_substrings"]).lower()
    for token in ("penis", "testis", "scrotum", "pubic hair"):
        assert token in deny


@pytest.mark.skipif(not SKIN_XML.exists(), reason="Run scripts/atlas/build_fullbody_skin.py first")
def test_fullbody_skin_xml_loads():
    assert SKIN_XML.suffix == ".xml"
    model = mujoco.MjModel.from_xml_path(str(SKIN_XML))
    assert model.nskin >= 1
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "hip_flexion_r")
    if jid >= 0:
        q0 = data.qpos.copy()
        data.qpos[model.jnt_qposadr[jid]] = q0[model.jnt_qposadr[jid]] + 0.4
        mujoco.mj_forward(model, data)
        assert np.all(np.isfinite(data.xpos))
    mujoco.mj_step(model, data)


@pytest.mark.skipif(not SKIN_XML.exists(), reason="Run scripts/atlas/build_fullbody_skin.py first")
def test_overlay_is_myofullbody_scale():
    model = mujoco.MjModel.from_xml_path(str(SKIN_XML))
    assert model.nbody > 80
    assert model.nu > 300


@pytest.mark.skipif(not SKIN_XML.exists(), reason="Run scripts/atlas/build_fullbody_skin.py first")
def test_fullbody_skin_xpose_pose():
    """X-pose kinematics: arms horizontal, hips abducted."""
    import sys

    sys.path.insert(0, str(SCRIPTS))
    from preview_fullbody_skin import apply_xpose

    model = mujoco.MjModel.from_xml_path(str(SKIN_XML))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    apply_xpose(model, data)
    assert np.all(np.isfinite(data.xpos))

    tal_r = data.xpos[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "talus_r")]
    tal_l = data.xpos[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "talus_l")]
    assert float(tal_l[0] - tal_r[0]) > 0.8

    rad_r = data.xpos[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "radius_r")]
    hum_r = data.xpos[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "humerus_r")]
    assert abs(float(rad_r[2] - hum_r[2])) < 0.12


@pytest.mark.skipif(not SKIN_XML.exists(), reason="Run scripts/atlas/build_fullbody_skin.py first")
def test_xpose_skin_quality():
    """X-pose metrics must catch bat-wings (stretch) and hand follow, not just bone gaps."""
    import sys

    sys.path.insert(0, str(SCRIPTS))
    from preview_fullbody_skin import xpose_quality_report

    model = mujoco.MjModel.from_xml_path(str(SKIN_XML))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    report = xpose_quality_report(model, data)

    bad_bones = [b for b, g in report["bone_gaps"].items() if not g["ok"]]
    bad_hands = [s for s, h in report["hands"].items() if not h["ok"]]
    bad_sleeves = [s for s, v in report["sleeves"].items() if not v["ok"]]
    msg = []
    st = report.get("stretch")
    if st is not None and not st["ok"]:
        msg.append(f"stretch: p99={st['p99']:.2f} n_gt5={st['n_gt5']} edge_p99={st['edge_p99']:.3f}")
    if bad_bones:
        msg.append("bone gaps: " + ", ".join(f"{b} p90={report['bone_gaps'][b]['p90']:.3f}" for b in bad_bones))
    if bad_hands:
        msg.append("hands: " + ", ".join(f"{s} err={report['hands'][s]['err']:.3f}" for s in bad_hands))
    if bad_sleeves:
        msg.append("sleeves: " + ", ".join(f"{s} p90={report['sleeves'][s]['p90']:.3f}" for s in bad_sleeves))
    assert report["ok"], "X-pose skin quality failed:\n  " + "\n  ".join(msg)
