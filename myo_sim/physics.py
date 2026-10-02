"""Shared musculoskeletal driver for MuJoCo C plugin and MuJoCo Warp backends.

Both backends consume the same muscle records / contact tables (see ``sprinter_assets``,
``fullbody_simple_assets``). The C path embeds ``myo.muscles`` and advances via ``mj_step``;
the Warp path applies ``MuscleSystem`` / ``ContactSystem`` around ``mjw.step``.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Literal


def limits_from_map(map_dict: dict) -> list[dict]:
    """Convert a map.json ``limits`` list (coord names) into MuscleSystem limit dicts."""
    qposadr = map_dict["qposadr"]
    dofadr = map_dict["dofadr"]
    out = []
    for lim in map_dict.get("limits", []):
        coord = lim["coord"]
        out.append(
            {
                "qposadr": int(qposadr[coord]),
                "dofadr": int(dofadr[coord]),
                "q_low": float(lim["q_low"]),
                "q_up": float(lim["q_up"]),
                "k_low": float(lim["k_low"]),
                "k_up": float(lim["k_up"]),
                "damping": float(lim["damping"]),
                "transition": float(lim["transition"]),
            }
        )
    return out


class MusculoskeletalDriver:
    """Thin wrapper around ``.muscle`` / ``.contact`` with shared prepare / step hooks."""

    def __init__(self, backend: Literal["c", "warp"], muscle, contact, *, nworld: int = 1):
        self.backend = backend
        self.muscle = muscle
        self.contact = contact
        self.nworld = int(nworld)
        self.mjm = None
        self.m = None  # mjw model when backend=warp; same as mjm when backend=c
        self.d = None
        self._tmpdir = None
        self._spec_path = None
        self._xml_path = None

    def prepare_model(
        self,
        xml_path: str | Path,
        *,
        spec_path: str | Path | None = None,
        out_xml: str | Path | None = None,
    ):
        """Load the model for this backend.

        Returns ``(m, d)``:
        - ``backend=\"c\"``: ``(MjModel, MjData)`` with plugin injected.
        - ``backend=\"warp\"``: ``(mjw.Model, mjw.Data)``; ``self.mjm`` holds the host MjModel.
        """
        xml_path = Path(xml_path)
        if self.backend == "c":
            return self._prepare_c(xml_path, spec_path=spec_path, out_xml=out_xml)
        return self._prepare_warp(xml_path)

    def _prepare_c(self, xml_path: Path, *, spec_path, out_xml):
        import mujoco

        from myo_sim.plugin import cmyo

        cmyo.ensure_built()
        cmyo.load_plugin()
        if self._tmpdir is None:
            self._tmpdir = tempfile.TemporaryDirectory(prefix="myo_sim_plugin_")
        tmp = Path(self._tmpdir.name)
        spec_path = Path(spec_path) if spec_path is not None else tmp / "muscles.spec"
        contact = None
        if self.contact is not None:
            contact = self.contact.as_contact_dict()
        self.muscle.write_spec(spec_path, contact=contact)
        self._spec_path = str(Path(spec_path).resolve())
        xml_text = xml_path.read_text()
        xml_text = cmyo.inject_plugin_xml(xml_text, self._spec_path)
        # Keep compiled XML next to the source model so relative meshdir/texturedir still resolve.
        out_path = Path(out_xml) if out_xml is not None else xml_path.with_name(xml_path.stem + "_plugin.xml")
        out_path.write_text(xml_text)
        self._xml_path = str(out_path)
        mjm = mujoco.MjModel.from_xml_path(self._xml_path)
        mjd = mujoco.MjData(mjm)
        self.mjm = mjm
        self.m = mjm
        self.d = mjd
        self.muscle.bind(mjm, mjd)
        if self.contact is not None:
            self.contact.bind(mjm, mjd, self.muscle.n)
        return mjm, mjd

    def _prepare_warp(self, xml_path: Path):
        import mujoco
        import mujoco_warp as mjw

        mjm = mujoco.MjModel.from_xml_path(str(xml_path))
        self.mjm = mjm
        self.m = mjw.put_model(mjm)
        self.d = mjw.put_data(mjm, mujoco.MjData(mjm), nworld=self.nworld)
        return self.m, self.d

    def before_step(self, m=None, d=None):
        """Warp: compute+apply muscles and contacts. C: no-op (plugin runs in ``mj_step``)."""
        m = self.m if m is None else m
        d = self.d if d is None else d
        if self.backend == "warp":
            self.muscle.compute(d)
            self.muscle.apply(d, m, zero=True)
            if self.contact is not None:
                d.xfrc_applied.zero_()
                self.contact.compute(m, d)
                self.contact.apply(d)
        # C: nothing

    def after_step(self, timestep: float | None = None):
        """Warp: advance muscle state. C: sync numpy views from plugin state."""
        if self.backend == "warp":
            dt = float(timestep if timestep is not None else (self.mjm.opt.timestep if self.mjm is not None else 0.0))
            self.muscle.advance(dt)
        else:
            self.muscle._sync_state()
            if self.contact is not None:
                self.contact.sync()

    def equilibrate(self, d=None, reset=None):
        d = self.d if d is None else d
        return self.muscle.equilibrate(d, reset=reset)

    def step(self, m=None, d=None, timestep: float | None = None):
        """One full physics step for the active backend."""
        m = self.m if m is None else m
        d = self.d if d is None else d
        if self.backend == "c":
            import mujoco

            mujoco.mj_step(m, d)
            self.after_step(timestep)
            return
        import mujoco_warp as mjw

        dt = float(timestep if timestep is not None else self.mjm.opt.timestep)
        self.muscle.compute(d)
        self.muscle.apply(d, m, zero=True)
        if self.contact is not None:
            d.xfrc_applied.zero_()
            self.contact.compute(m, d)
            self.contact.apply(d)
        mjw.step(m, d)
        self.muscle.advance(dt)


def make_musculoskeletal(
    backend: Literal["c", "warp"],
    muscles,
    contact=None,
    *,
    nworld: int = 1,
    limits=(),
    device=None,
    met_flags: int = 15,
    aerobic: float = 1.0,
    effort: float = 1.0,
    nbody: int | None = None,
) -> MusculoskeletalDriver:
    """Build a dual-backend driver from shared muscle / contact assets.

    Parameters
    ----------
    backend:
        ``\"c\"`` — MuJoCo float64 passive plugin. ``\"warp\"`` — float32 Warp kernels.
    muscles:
        List of muscle record dicts (same format for both backends).
    contact:
        Optional dict with ``geom_flat``, ``pairs``, ``naconmax`` (from ``load_contact()``).
    nbody:
        Required when ``contact`` is set and ``backend=\"warp\"`` (``ContactSystem`` needs it).
        For ``backend=\"c\"`` it is optional (defaults to 0 if omitted).
    """
    backend = backend.lower()  # type: ignore[assignment]
    if backend not in ("c", "warp"):
        raise ValueError(f"backend must be 'c' or 'warp', got {backend!r}")

    if backend == "c":
        from myo_sim.plugin.contact_system import ContactSystem as CContact
        from myo_sim.plugin.muscle_system import MuscleSystem as CMuscle

        con = None
        if contact is not None:
            con = CContact(
                contact["geom_flat"],
                contact["pairs"],
                nworld,
                nbody if nbody is not None else 0,
                naconmax=int(contact["naconmax"]),
            )
        mus = CMuscle(
            muscles,
            nworld,
            met_flags=met_flags,
            aerobic=aerobic,
            effort=effort,
            limits=limits,
            contact=con.as_contact_dict() if con is not None else None,
        )
        return MusculoskeletalDriver("c", mus, con, nworld=nworld)

    from myo_sim.warp.contact_system import ContactSystem as WContact
    from myo_sim.warp.muscle_system import MuscleSystem as WMuscle

    mus = WMuscle(muscles, nworld, met_flags=met_flags, aerobic=aerobic, effort=effort, device=device, limits=limits)
    con = None
    if contact is not None:
        if nbody is None:
            raise ValueError("nbody is required when backend='warp' and contact is provided")
        con = WContact(
            contact["geom_flat"],
            contact["pairs"],
            nworld,
            nbody,
            naconmax=int(contact["naconmax"]),
            device=device,
        )
    return MusculoskeletalDriver("warp", mus, con, nworld=nworld)
