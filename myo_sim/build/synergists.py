"""Opt-in deep hip muscles (pectineus, quadratus femoris, gemelli) added to a composed leg model.

Geometry and strength come from the OpenSim gait2392 model (Delp et al. 1990; Seth, Anderson, Delp;
CC BY 3.0). Pelvis points were mapped into the Rajagopal pelvis frame used here with a similarity
transform fitted on 27 points of 21 muscles shared by both models (leave-one-muscle-out error about
11 mm); femur points are shared between the models. Right-side positions are listed, the left side
mirrors z. ``range`` is the normalised fibre length over a gait-scale hip range of motion
(flexion -20..50, adduction +-15, rotation +-20 deg), i.e. (path length - tendon slack) / optimal fibre length.
"""

from __future__ import annotations

from dataclasses import dataclass

import mujoco

_REFERENCE = "piri"


@dataclass(frozen=True)
class HipMuscle:
    name: str
    pelvis_pos: tuple[float, float, float]
    femur_pos: tuple[float, float, float]
    tendon_slack: float
    force: float
    fibre_range: tuple[float, float]
    lengthrange: tuple[float, float]


HIP_SYNERGISTS = (
    HipMuscle(
        "pect", (-0.0261, -0.0808, 0.0455), (-0.0122, -0.0822, 0.0253), 0.033, 266.0, (0.423016, 0.870361), (0.075302, 0.120036)
    ),
    HipMuscle(
        "quad_fem",
        (-0.0896, -0.1251, 0.0455),
        (-0.0381, -0.0359, 0.0366),
        0.024,
        381.0,
        (0.376038, 1.313616),
        (0.044306, 0.094935),
    ),
    HipMuscle(
        "gem", (-0.0942, -0.0936, 0.0637), (-0.0142, -0.0033, 0.0443), 0.039, 164.0, (0.442074, 1.585933), (0.049610, 0.077062)
    ),
)


def add_hip_synergists(spec: mujoco.MjSpec) -> mujoco.MjSpec:
    """Add pectineus, quadratus femoris and gemelli (both sides) to ``spec`` in place.

    The muscles are appended after the existing actuators, so existing action indices do not change.

    Args:
        spec: Composed, uncompiled MjSpec with ``pelvis`` and ``femur_r/_l`` bodies and the piriformis
            muscles, whose actuator settings are reused.

    Returns:
        The same spec.
    """
    if spec.body("pelvis") is None:
        raise KeyError("spec has no pelvis body")
    for side, sign in (("r", 1.0), ("l", -1.0)):
        if spec.body(f"femur_{side}") is None or spec.site(f"{_REFERENCE}-P1_{side}") is None:
            raise KeyError(f"spec lacks femur_{side} or the {_REFERENCE} reference muscle")
        # copy reference settings first: MjSpec handles can dangle once elements are added
        site_ref = spec.site(f"{_REFERENCE}-P1_{side}")
        site_style = {"size": list(site_ref.size), "group": int(site_ref.group), "rgba": list(site_ref.rgba)}
        ten_ref = next(t for t in spec.tendons if t.name == f"{_REFERENCE}_{side}_tendon")
        ten_style = {"width": float(ten_ref.width), "rgba": list(ten_ref.rgba), "group": int(ten_ref.group)}
        act_ref = next(a for a in spec.actuators if a.name == f"{_REFERENCE}_{side}")
        act_style = {
            k: getattr(act_ref, k)
            for k in ("trntype", "gaintype", "biastype", "dyntype", "ctrllimited", "forcelimited", "group", "classname")
        }
        act_style.update({k: list(getattr(act_ref, k)) for k in ("dynprm", "ctrlrange", "forcerange")})
        for m in HIP_SYNERGISTS:
            for i, (body, pos) in enumerate((("pelvis", m.pelvis_pos), (f"femur_{side}", m.femur_pos)), 1):
                site = spec.body(body).add_site(name=f"{m.name}-P{i}_{side}")
                site.pos = [pos[0], pos[1], sign * pos[2]]
                for k, v in site_style.items():
                    setattr(site, k, v)
            tendon = spec.add_tendon(name=f"{m.name}_{side}_tendon", springlength=[m.tendon_slack, m.tendon_slack])
            tendon.wrap_site(f"{m.name}-P1_{side}")
            tendon.wrap_site(f"{m.name}-P2_{side}")
            for k, v in ten_style.items():
                setattr(tendon, k, v)
            act = spec.add_actuator(name=f"{m.name}_{side}", target=f"{m.name}_{side}_tendon")
            for k, v in act_style.items():
                setattr(act, k, v)
            prm = [m.fibre_range[0], m.fibre_range[1], m.force, 1.0, 0.05, 3.5, 10.0, 1.5, 1.4, 0.0]
            act.gainprm, act.biasprm = prm, prm
            act.lengthrange = list(m.lengthrange)
    return spec
