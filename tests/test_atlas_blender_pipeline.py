"""Smoke test for Blender atlas fit→rig pipeline (skips if Blender missing)."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import mujoco
import pytest

REPO = Path(__file__).resolve().parents[1]
BLENDER_DIR = REPO / "scripts" / "atlas" / "blender"
SKIN_XML = REPO / "myo_sim" / "models" / "visual" / "atlas" / "myofullbody_atlas_skin.xml"


def _blender_available() -> bool:
    sys.path.insert(0, str(BLENDER_DIR))
    try:
        from pipeline_paths import blender_bin

        return blender_bin().exists()
    except Exception:
        return False


@pytest.mark.skipif(not _blender_available(), reason="Blender binary not found")
def test_blender_pipeline_smoke():
    """export → step1 → step2 → import; overlay loads and X-pose stretch gate passes."""
    cmd = [sys.executable, str(BLENDER_DIR / "run_pipeline.py")]
    subprocess.run(cmd, check=True, cwd=str(REPO))
    assert SKIN_XML.exists()
    model = mujoco.MjModel.from_xml_path(str(SKIN_XML))
    assert model.nskin >= 1
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    sys.path.insert(0, str(REPO / "scripts" / "atlas"))
    from preview_fullbody_skin import xpose_quality_report

    report = xpose_quality_report(model, data)
    # Plan gate: overlay loads + stretch-aware X-pose metrics (not full sleeve/palm advisory).
    assert report["stretch"]["ok"], f"stretch failed: {report['stretch']}"
