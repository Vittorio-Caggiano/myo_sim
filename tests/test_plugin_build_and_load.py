"""Build and load the myo.muscles C plugin shared library."""

from __future__ import annotations

import pytest


def test_ensure_built_and_load_plugin():
    pytest.importorskip("mujoco")
    from myo_sim.plugin import ensure_built, load_plugin

    try:
        path = ensure_built()
    except RuntimeError as e:
        pytest.skip(str(e))
    assert path.exists()
    load_plugin()
    # second call is a no-op
    load_plugin()


def test_plugin_public_exports():
    from myo_sim import plugin

    for name in (
        "MuscleSystem",
        "ContactSystem",
        "ensure_built",
        "load_plugin",
        "write_spec",
        "plugin_xml",
        "state_view",
        "contact_state_view",
    ):
        assert hasattr(plugin, name), name
