"""Companion (non-XML) data for the `myosprinter` fragment: per-muscle DGF parameter records for
myo_sim.warp.MuscleSystem / myo_sim.plugin write_spec, contact geometry for
myo_sim.warp.ContactSystem, and the qpos/dof/muscle name maps used to align them with the compiled model.

Derived from Bolt's example full-body model (data/models/example_model.osim) and the matching
msk_envs sprinter assets (msk_models/sprinter/), both Apache-2.0; see NOTICE.

Both backends consume the same records: Warp via MuscleSystem/ContactSystem constructors; MuJoCo C via
``write_plugin_spec`` (or ``myo_sim.physics.make_musculoskeletal(backend=\"c\", ...)``).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

_DIR = Path(__file__).resolve().parent / "models" / "sprint"


def load_muscles() -> list:
    """Per-muscle parameter records (list of dicts), MuJoCo muscle order -- feed directly to
    myo_sim.warp.MuscleSystem(muscles, nworld, ...) or myo_sim.plugin.cmyo.write_spec(..., muscles, ...)."""
    return list(np.load(_DIR / "myosprinter_muscles.npy", allow_pickle=True))


def load_contact() -> dict:
    """geom_flat (ngeom, 26+), pairs (npair, 2), naconmax -- feed to myo_sim.warp.ContactSystem(geom_flat,
    pairs, nworld, nbody, naconmax=naconmax) or myo_sim.plugin.cmyo.write_spec(..., contact=load_contact())."""
    cz = np.load(_DIR / "myosprinter_contact.npz")
    return {"geom_flat": cz["geom_flat"], "pairs": cz["pairs"], "naconmax": int(cz["naconmax"])}


def load_map() -> dict:
    """qposadr/dofadr/muscle- and actuator-order maps tying the compiled myosprinter model's indices to the
    muscle-record and contact-geometry orders above."""
    return json.loads((_DIR / "myosprinter_map.json").read_text())


def load_limits() -> list[dict]:
    """Joint-limit force records for MuscleSystem / write_spec (qposadr/dofadr resolved from the map)."""
    from myo_sim.physics import limits_from_map

    return limits_from_map(load_map())


def write_plugin_spec(path, *, limits=None, contact=True, met_flags=15, aerobic=1.0, effort=1.0) -> str:
    """Write a ``myo.muscles`` spec from packaged sprinter assets (same records as the Warp path).

    Parameters
    ----------
    contact:
        If True, include ``load_contact()``; if False, omit contacts; if a dict, use it.
    """
    from myo_sim.plugin import cmyo

    if limits is None:
        limits = load_limits()
    if contact is True:
        contact = load_contact()
    elif contact is False:
        contact = None
    path = str(path)
    cmyo.write_spec(
        path,
        load_muscles(),
        met_flags=met_flags,
        aerobic=aerobic,
        effort=effort,
        limits=limits,
        contact=contact,
    )
    return path


def xml_path() -> Path:
    """Path to the packaged noplugin-style myosprinter MJCF."""
    return _DIR / "myosprinter.xml"
