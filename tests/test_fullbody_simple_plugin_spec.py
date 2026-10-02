"""Smoke: write_plugin_spec from packaged assets produces a loadable myo.muscles instance."""

from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture(scope="module")
def plugin_ready():
    pytest.importorskip("mujoco")
    from myo_sim.plugin import ensure_built, load_plugin

    try:
        ensure_built()
    except RuntimeError as e:
        pytest.skip(str(e))
    load_plugin()


def test_sprinter_write_plugin_spec_loads(plugin_ready, tmp_path):
    import mujoco

    from myo_sim import sprinter_assets
    from myo_sim.plugin import cmyo

    spec = tmp_path / "sprinter.spec"
    sprinter_assets.write_plugin_spec(spec, contact=True)
    assert spec.is_file() and spec.stat().st_size > 100

    xml = sprinter_assets.xml_path().read_text()
    xml = cmyo.inject_plugin_xml(xml, str(spec.resolve()))
    out = tmp_path / "myosprinter_plugin.xml"
    out.write_text(xml)
    m = mujoco.MjModel.from_xml_path(str(out))
    d = mujoco.MjData(m)
    assert m.nplugin >= 1
    mujoco.mj_forward(m, d)
    st = cmyo.state_view(m, d, len(sprinter_assets.load_muscles()))
    assert st.shape[0] == len(sprinter_assets.load_muscles())
    assert st.shape[1] == cmyo.NSTATE


def test_fullbody_simple_write_plugin_spec(plugin_ready, tmp_path):
    from myo_sim import fullbody_simple_assets

    assets = Path(fullbody_simple_assets.__file__).resolve().parent / "models" / "fullbody_simple"
    if not (assets / "muscles.npy").is_file():
        pytest.skip("fullbody_simple assets not packaged")
    spec = tmp_path / "fullbody_simple.spec"
    fullbody_simple_assets.write_plugin_spec(spec, contact=True)
    assert spec.is_file() and spec.stat().st_size > 100
    text = spec.read_text().splitlines()
    # header: nmuscles met aerobic effort
    nmus = int(text[0].split()[0])
    assert nmus == len(fullbody_simple_assets.load_muscles())
