"""CPU / MuJoCo-plugin MuscleSystem facade (Warp-shaped API).

Forces, activation dynamics, joint limits and (optional) Hunt–Crossley contacts are applied
inside ``mujoco.mj_step`` / ``mj_forward`` by the ``myo.muscles`` passive plugin. ``compute`` /
``apply`` / ``advance`` are therefore no-ops; write a plugin spec, inject it into the MJCF, load
the plugin library, then step MuJoCo normally.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from . import cmyo


class MuscleSystem:
    """Host-side muscle records + plugin state view, mirroring ``myo_sim.warp.MuscleSystem`` ctor args."""

    def __init__(
        self,
        muscles,
        nworld: int = 1,
        met_flags: int = 15,
        aerobic: float = 1.0,
        effort: float = 1.0,
        device=None,  # noqa: ARG002 — API parity with Warp (unused on CPU)
        limits=(),
        contact=None,
    ):
        self.muscles = list(muscles)
        self.n = len(self.muscles)
        self.nworld = int(nworld)
        self.met = (int(met_flags), float(aerobic), float(effort))
        self.limits = list(limits)
        self.contact = contact
        self.params = np.stack([np.asarray(m["p"], dtype=np.float64) for m in self.muscles])
        self._mjm = None
        self._mjd = None
        self._instance = 0
        self.spec_path: str | None = None
        # numpy stand-in for Warp's (nworld, n, NSTATE) state buffer; filled after bind + forward
        self.state = np.zeros((self.nworld, self.n, cmyo.NSTATE), dtype=np.float64)

    def write_spec(self, path, contact=None, met_flags=None, aerobic=None, effort=None, limits=None):
        """Write the binary-free text spec consumed by ``myo.muscles``."""
        path = str(path)
        cmyo.write_spec(
            path,
            self.muscles,
            met_flags=self.met[0] if met_flags is None else met_flags,
            aerobic=self.met[1] if aerobic is None else aerobic,
            effort=self.met[2] if effort is None else effort,
            limits=self.limits if limits is None else limits,
            contact=self.contact if contact is None else contact,
        )
        self.spec_path = path
        return path

    def bind(self, mjm, mjd, instance: int = 0):
        """Attach to a compiled model/data that already embeds the plugin instance."""
        self._mjm = mjm
        self._mjd = mjd
        self._instance = int(instance)
        self._sync_state()

    def _sync_state(self):
        if self._mjm is None or self._mjd is None:
            return
        st = cmyo.state_view(self._mjm, self._mjd, self.n, self._instance)
        # plugin is single-world; broadcast to nworld slot 0 for API shape parity
        self.state[0] = st

    def compute(self, d=None):  # noqa: ARG002
        """No-op: the plugin evaluates muscles during ``mj_forward`` / ``mj_step``."""
        return None

    def apply(self, d=None, m=None, zero=True):  # noqa: ARG002
        """No-op: the plugin writes to ``qfrc_passive`` inside MuJoCo."""
        return None

    def advance(self, timestep=None):  # noqa: ARG002
        """No-op: activation / fiber state advance inside the plugin during ``mj_step``."""
        return None

    def equilibrate(self, d=None, reset=None):
        """Re-solve fiber length at force equilibrium into plugin state.

        Requires ``bind(mjm, mjd)`` first (or pass ``d`` as MjData with a prior bind for the model).
        ``reset`` is a bool per world (None = all); only world 0 is meaningful for the C plugin.
        """
        if self._mjm is None:
            raise RuntimeError("MuscleSystem.bind(mjm, mjd) required before equilibrate()")
        mjd = self._mjd if d is None else d
        import mujoco

        mujoco.mj_forward(self._mjm, mjd)
        st = cmyo.state_view(self._mjm, mjd, self.n, self._instance)
        if reset is None:
            do = True
        else:
            reset_arr = np.asarray(reset, dtype=bool).reshape(-1)
            do = bool(reset_arr[0]) if reset_arr.size else True
        if do:
            ln = cmyo.equilibrium(st[:, 3].copy(), st[:, 4].copy(), st[:, 0].copy(), self.params)
            st[:, 1] = ln
            # Re-evaluate plugin so force / act_dot channels match the new fiber lengths.
            mujoco.mj_forward(self._mjm, mjd)
        self._mjd = mjd
        self._sync_state()
        return cmyo.state_view(self._mjm, mjd, self.n, self._instance)

    def step(self, m, d, timestep=None):  # noqa: ARG002
        """``mj_step`` convenience (plugin owns muscle/contact inside the step)."""
        import mujoco

        mujoco.mj_step(m, d)
        self.bind(m, d, self._instance)

    def attach_to_xml(self, xml_text: str, spec_path: str | Path, contact=None) -> str:
        """Write the plugin spec and return MJCF text with the extension injected."""
        self.write_spec(spec_path, contact=contact)
        return cmyo.inject_plugin_xml(xml_text, str(spec_path))
