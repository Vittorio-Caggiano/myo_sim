"""Render HM08 skin overlays on fullbody and body-segment models.

Usage:
    uv run python tests/render_hm08_skin.py
"""

from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np

from myo_sim import load_spec
from myo_sim.skin import attach_skin

OUT_DIR = Path(__file__).resolve().parent / "output" / "hm08_skin"
WIDTH, HEIGHT = 900, 900


def _load(model_name: str, segment: str | None, skin_name: str):
    spec = load_spec(model_name)
    spec.visual.global_.offwidth = max(WIDTH, int(spec.visual.global_.offwidth))
    spec.visual.global_.offheight = max(HEIGHT, int(spec.visual.global_.offheight))
    attach_skin(spec, segment=segment, name=skin_name)
    model = spec.compile()
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    return model, data


TARGETS: list[tuple[str, str | None, str, dict]] = [
    ("myofullbody", None, "fullbody_rest", {"azimuth": 140, "elevation": -12, "distance_scale": 2.2}),
    (
        "myofullbody",
        None,
        "fullbody_raised_arm",
        {"pose": {"shoulder_elv_r": 1.8}, "azimuth": 140, "elevation": -8, "distance_scale": 2.2},
    ),
    ("myoarm_r", "arm_r", "arm_r", {"azimuth": 120, "elevation": -15, "distance_scale": 2.4}),
    ("myolegs", "legs", "legs", {"azimuth": 130, "elevation": -10, "distance_scale": 2.3}),
    ("myotorso", "torso", "torso", {"azimuth": 145, "elevation": -8, "distance_scale": 2.5}),
]


def _set_hinge(model: mujoco.MjModel, data: mujoco.MjData, name: str, value: float) -> None:
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
    if jid < 0:
        return
    lo, hi = model.jnt_range[jid]
    data.qpos[model.jnt_qposadr[jid]] = float(np.clip(value, lo, hi))


def _character_bounds(model: mujoco.MjModel, data: mujoco.MjData) -> tuple[np.ndarray, float]:
    mins = np.full(3, np.inf)
    maxs = np.full(3, -np.inf)
    for gid in range(model.ngeom):
        if model.geom_bodyid[gid] == 0:
            continue
        pos = data.geom_xpos[gid]
        size = float(model.geom_size[gid].max()) + 0.05
        mins = np.minimum(mins, pos - size)
        maxs = np.maximum(maxs, pos + size)
    if not np.all(np.isfinite(mins)):
        return model.stat.center.copy(), float(model.stat.extent)
    center = (mins + maxs) / 2.0
    extent = float(np.linalg.norm(maxs - mins)) / 2.0
    return center, max(extent, 0.2)


def render_one(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    out_path: Path,
    *,
    azimuth: float,
    elevation: float,
    distance_scale: float,
) -> None:
    center, extent = _character_bounds(model, data)
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = center
    cam.distance = distance_scale * extent
    cam.azimuth = azimuth
    cam.elevation = elevation

    opt = mujoco.MjvOption()
    opt.sitegroup[:] = False
    # Prefer the skin; hide wrap (3) and collision (4) geoms.
    for g in range(6):
        opt.geomgroup[g] = g not in (3, 4)
    opt.flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT] = False

    with mujoco.Renderer(model, height=HEIGHT, width=WIDTH) as renderer:
        renderer.update_scene(data, camera=cam, scene_option=opt)
        pixels = renderer.render()

    from PIL import Image

    Image.fromarray(pixels).save(out_path)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for model_name, segment, label, cfg in TARGETS:
        model, data = _load(model_name, segment, f"hm08_{label}")
        for joint, value in cfg.get("pose", {}).items():
            _set_hinge(model, data, joint, value)
        mujoco.mj_forward(model, data)
        out = OUT_DIR / f"hm08_{label}.png"
        render_one(
            model,
            data,
            out,
            azimuth=cfg["azimuth"],
            elevation=cfg["elevation"],
            distance_scale=cfg["distance_scale"],
        )
        print(f"Wrote {out} (nskinvert={model.nskinvert})")


if __name__ == "__main__":
    main()
