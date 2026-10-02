"""CPU / MuJoCo-plugin ContactSystem facade (Warp-shaped API).

Hunt–Crossley contacts are baked into the ``myo.muscles`` plugin spec (``contact=`` dict) and
evaluated inside ``mj_forward`` / ``mj_step``. ``compute`` / ``apply`` are no-ops; read GRF from
plugin state after a forward step.
"""

from __future__ import annotations

import numpy as np

from . import cmyo


class ContactSystem:
    """Contact geometry records + plugin GRF view, mirroring ``myo_sim.warp.ContactSystem`` ctor args."""

    def __init__(self, geom_flat, pairs, nworld, nbody, naconmax=512, device=None):  # noqa: ARG002
        self.geom_flat = np.asarray(geom_flat, dtype=np.float64)
        self.pairs = np.asarray(pairs, dtype=np.int32).reshape(-1, 2)
        self.nworld = int(nworld)
        self.nbody = int(nbody)
        self.ngeom = int(self.geom_flat.shape[0])
        self.naconmax = int(naconmax)
        self._mjm = None
        self._mjd = None
        self._n_muscles = 0
        self._instance = 0
        self.grf = np.zeros((self.nworld, 3), dtype=np.float64)
        self.nacon = np.zeros(self.nworld, dtype=np.int32)
        self.ncollision = np.zeros(self.nworld, dtype=np.int32)

    def as_contact_dict(self) -> dict:
        """Shape expected by ``cmyo.write_spec(..., contact=...)``."""
        return {"geom_flat": self.geom_flat, "pairs": self.pairs, "naconmax": self.naconmax}

    def bind(self, mjm, mjd, n_muscles: int, instance: int = 0):
        self._mjm = mjm
        self._mjd = mjd
        self._n_muscles = int(n_muscles)
        self._instance = int(instance)
        self.sync()

    def sync(self):
        """Pull GRF / contact counts from plugin state into ``grf`` / ``nacon`` / ``ncollision``."""
        if self._mjm is None or self._mjd is None:
            return
        cs = cmyo.contact_state_view(self._mjm, self._mjd, self._n_muscles, self._instance)
        self.grf[0] = cs[:3]
        self.nacon[0] = int(cs[3])
        self.ncollision[0] = int(cs[4])

    def compute(self, m=None, d=None):  # noqa: ARG002
        """No-op: contacts run inside the plugin during ``mj_forward`` / ``mj_step``."""
        if d is not None and self._mjm is not None:
            self._mjd = d
            self.sync()
        return None

    def apply(self, d=None):  # noqa: ARG002
        """No-op: the plugin applies contact wrenches via ``mj_applyFT``."""
        return None

    def run(self):
        """No-op peer of Warp ``ContactSystem.run`` (geometry already in plugin spec)."""
        self.sync()
