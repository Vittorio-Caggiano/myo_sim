"""MuJoCo C plugin ``myo.muscles`` (CPU) — De Groote–Fregly muscles + Hunt–Crossley contacts.

Not imported automatically by ``import myo_sim``. Build the shared libraries first
(``ensure_built()`` or ``plugin/build.sh``; requires a C compiler and the ``mujoco`` Python
package) before ``load_plugin()``.

Optional Warp peer lives in ``myo_sim.warp``; use ``myo_sim.physics.make_musculoskeletal`` to
switch backends on the same muscle/contact assets.
"""

from __future__ import annotations

from myo_sim.plugin.cmyo import (
    MET_BHARGAVA,
    MET_BOLT,
    MET_DEFAULT,
    MET_FORBID_NEG,
    MET_MIN_HEAT,
    MET_NEG_WORK,
    NSTATE,
    PLUGIN_LIB,
    contact_state_view,
    ensure_built,
    inject_plugin_xml,
    load_plugin,
    plugin_xml,
    state_view,
    write_spec,
)
from myo_sim.plugin.contact_system import ContactSystem
from myo_sim.plugin.muscle_system import MuscleSystem

__all__ = [
    "ContactSystem",
    "MET_BHARGAVA",
    "MET_BOLT",
    "MET_DEFAULT",
    "MET_FORBID_NEG",
    "MET_MIN_HEAT",
    "MET_NEG_WORK",
    "MuscleSystem",
    "NSTATE",
    "PLUGIN_LIB",
    "contact_state_view",
    "ensure_built",
    "inject_plugin_xml",
    "load_plugin",
    "plugin_xml",
    "state_view",
    "write_spec",
]
