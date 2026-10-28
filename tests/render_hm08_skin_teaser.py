"""HM08 skin teaser: MyoSuite MuscleMimic motions with skin fade in/out.

Records an MP4 via MuJoCo offscreen rendering (skins are not supported in
mjviser yet): black backdrop, subject-centered camera starting from the front.
Optional ``--mjviser`` opens an interactive trajectory scrubber for the same
concatenated motion (skeleton / geom view).

Usage:
    uv run python tests/render_hm08_skin_teaser.py
    uv run python tests/render_hm08_skin_teaser.py --mjviser
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np

from myo_sim import load_spec
from myo_sim.skin import attach_skin

REPO = Path(__file__).resolve().parents[1]
OUT_DIR = REPO / "tests" / "output" / "hm08_skin"
DEFAULT_CACHE = Path.home() / ".musclemimic" / "caches" / "AMASS" / "MyoFullBody"

# MuscleMimic / MyoSuite demo clips under the local AMASS cache.
DEFAULT_CLIPS = (
    "gmr/KIT/425/walking_medium01_poses.npz",
    "gmr/KIT/425/turn_right02_poses.npz",
    "gmr/Transitions_mocap/mazen_c3d/jumpingjacks_jumpingtwist360_poses.npz",
)

JOINT_ALIASES = {
    "elbow_flex_r": "elbow_flexion_r",
    "elbow_flex_l": "elbow_flexion_l",
}

WIDTH, HEIGHT = 1280, 720
FPS = 30

FADE_IN_S = 2.8
FADE_OUT_S = 3.0
OPEN_HOLD_S = 0.8
BETWEEN_FADE_S = 2.4
BETWEEN_HOLD_S = 0.15

# Camera: 0° = front (+Y in MyoFullBody), then a slow orbit at fixed distance.
CAM_AZIMUTH_START = 0.0
CAM_AZIMUTH_SPAN = 90.0
CAM_ELEVATION = -8.0
CAM_DISTANCE = 3.0


@dataclass
class ClipSegment:
    label: str
    qpos: np.ndarray  # (T, nq)
    frequency_hz: float


def _resolve_clip(path: str, cache: Path) -> Path:
    p = Path(path).expanduser()
    if p.is_file():
        return p.resolve()
    cached = cache / path
    if cached.is_file():
        return cached.resolve()
    raise FileNotFoundError(f"Motion not found: {path}\nExpected under {cache} (MyoSuite MuscleMimic demo cache).")


def _joint_width(model: mujoco.MjModel, joint_name: str) -> int:
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
    if jid < 0:
        raise KeyError(joint_name)
    jtype = int(model.jnt_type[jid])
    if jtype == int(mujoco.mjtJoint.mjJNT_FREE):
        return 7
    if jtype == int(mujoco.mjtJoint.mjJNT_BALL):
        return 4
    return 1


def load_clip_qpos(model: mujoco.MjModel, path: Path) -> ClipSegment:
    """Map a MyoSuite motion NPZ onto ``model.qpos`` by joint name."""
    z = np.load(path, allow_pickle=True)
    if "qpos" not in z.files or "joint_names" not in z.files:
        raise ValueError(f"{path} missing qpos/joint_names")
    src = np.asarray(z["qpos"], dtype=np.float64)
    names = [JOINT_ALIASES.get(str(n), str(n)) for n in z["joint_names"]]
    freq = float(z["frequency"]) if "frequency" in z.files else 100.0

    out = np.repeat(np.asarray(model.qpos0, dtype=np.float64)[None, :], src.shape[0], axis=0)
    cursor = 0
    for name in names:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if jid < 0:
            raise KeyError(f"Unknown joint in motion clip: {name} ({path.name})")
        width = _joint_width(model, name)
        adr = int(model.jnt_qposadr[jid])
        out[:, adr : adr + width] = src[:, cursor : cursor + width]
        cursor += width
    if cursor != src.shape[1]:
        raise ValueError(f"qpos width mismatch in {path}: consumed {cursor}, have {src.shape[1]}")
    return ClipSegment(label=path.stem, qpos=out, frequency_hz=freq)


def subsample(qpos: np.ndarray, src_hz: float, dst_fps: float, max_seconds: float) -> np.ndarray:
    step = max(1, int(round(src_hz / dst_fps)))
    frames = qpos[::step]
    max_n = max(1, int(round(max_seconds * dst_fps)))
    return frames[:max_n]


def smoothstep(t: float) -> float:
    t = float(np.clip(t, 0.0, 1.0))
    return t * t * (3.0 - 2.0 * t)


def smootherstep(t: float) -> float:
    """Ken Perlin's smootherstep — gentler ease than smoothstep."""
    t = float(np.clip(t, 0.0, 1.0))
    return t * t * t * (t * (t * 6.0 - 15.0) + 10.0)


def _fill_texture(model: mujoco.MjModel, tex_id: int, rgb: np.ndarray) -> None:
    """Write an (H, W, 3) uint8 image into ``model.tex_data`` for ``tex_id``."""
    w = int(model.tex_width[tex_id])
    h = int(model.tex_height[tex_id])
    nch = int(model.tex_nchannel[tex_id])
    adr = int(model.tex_adr[tex_id])
    flat = np.asarray(rgb, dtype=np.uint8).reshape(h, w, -1)
    if flat.shape[2] != nch:
        raise ValueError(f"texture channels {flat.shape[2]} != {nch}")
    model.tex_data[adr : adr + h * w * nch] = flat.reshape(-1)


def apply_black_look(model: mujoco.MjModel) -> None:
    """Black sky + visible dark grid floor, soft headlight on the subject."""
    model.vis.rgba.haze[:] = (0.0, 0.0, 0.0, 0.0)
    model.vis.headlight.ambient[:] = (0.35, 0.35, 0.35)
    model.vis.headlight.diffuse[:] = (0.65, 0.65, 0.65)
    model.vis.headlight.specular[:] = (0.15, 0.15, 0.15)

    sky_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_TEXTURE, "sky")
    if sky_id >= 0:
        w = int(model.tex_width[sky_id])
        h = int(model.tex_height[sky_id])
        _fill_texture(model, sky_id, np.zeros((h, w, 3), dtype=np.uint8))

    # Dark grid floor that still reads against the black backdrop.
    plane_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_TEXTURE, "texplane")
    if plane_id >= 0:
        w = int(model.tex_width[plane_id])
        h = int(model.tex_height[plane_id])
        yy, xx = np.mgrid[0:h, 0:w]
        cell = 8
        line = ((xx % cell) == 0) | ((yy % cell) == 0)
        img = np.empty((h, w, 3), dtype=np.uint8)
        img[:] = (28, 28, 30)
        img[line] = (55, 56, 60)
        _fill_texture(model, plane_id, img)

    mat_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_MATERIAL, "MatPlane")
    if mat_id >= 0:
        model.mat_rgba[mat_id] = (1.0, 1.0, 1.0, 1.0)
        model.mat_reflectance[mat_id] = 0.05
        model.mat_shininess[mat_id] = 0.02
    floor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    if floor_id >= 0:
        model.geom_rgba[floor_id] = (1.0, 1.0, 1.0, 1.0)


def build_model_with_skin():
    spec = load_spec("myofullbody")
    spec.visual.global_.offwidth = max(WIDTH, int(spec.visual.global_.offwidth))
    spec.visual.global_.offheight = max(HEIGHT, int(spec.visual.global_.offheight))
    attach_skin(spec, name="hm08_skin", rgba=(0.82, 0.64, 0.52, 1.0), group=2)
    model = spec.compile()
    apply_black_look(model)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    return model, data


def _ease_io(t: float) -> float:
    """Quintic ease-in-out — softer shoulders than smootherstep at the ends."""
    t = float(np.clip(t, 0.0, 1.0))
    if t < 0.5:
        return 16.0 * t * t * t * t * t
    u = 1.0 - t
    return 1.0 - 16.0 * u * u * u * u * u


def compose_skin_alpha(clip_lengths: list[int]) -> np.ndarray:
    """Long ease-in/out skin alpha across the full concatenated timeline."""
    n = int(sum(clip_lengths))
    alpha = np.ones(n, dtype=np.float64)
    open_hold = int(OPEN_HOLD_S * FPS)
    fade_in = int(FADE_IN_S * FPS)
    fade_out = int(FADE_OUT_S * FPS)
    between_fade = int(BETWEEN_FADE_S * FPS)
    between_hold = int(BETWEEN_HOLD_S * FPS)

    # Opening: skeleton hold → slow fade in.
    alpha[:open_hold] = 0.0
    for j in range(fade_in):
        idx = open_hold + j
        if idx < n:
            alpha[idx] = _ease_io(j / max(1, fade_in - 1))

    # Between clips: fade out → brief skeleton beat → fade in (overlap-safe mins/maxs).
    starts = np.cumsum([0, *clip_lengths[:-1]])
    for boundary in starts[1:]:
        out0 = boundary - between_fade - between_hold // 2
        for j in range(between_fade):
            idx = out0 + j
            if 0 <= idx < n:
                alpha[idx] = min(alpha[idx], 1.0 - _ease_io(j / max(1, between_fade - 1)))
        hold0 = boundary - between_hold // 2
        for j in range(between_hold):
            idx = hold0 + j
            if 0 <= idx < n:
                alpha[idx] = 0.0
        in0 = boundary + between_hold // 2
        for j in range(between_fade):
            idx = in0 + j
            if 0 <= idx < n:
                alpha[idx] = max(alpha[idx], _ease_io(j / max(1, between_fade - 1)))

    # Ending: slow fade out to skeleton.
    for j in range(fade_out):
        idx = n - fade_out + j
        if 0 <= idx < n:
            alpha[idx] = min(alpha[idx], 1.0 - _ease_io(j / max(1, fade_out - 1)))
    return np.clip(alpha, 0.0, 1.0)


def pin_root_xy(model: mujoco.MjModel, qpos: np.ndarray) -> np.ndarray:
    """Remove free-joint horizontal translation so the subject walks in place."""
    out = np.asarray(qpos, dtype=np.float64).copy()
    for jid in range(model.njnt):
        if int(model.jnt_type[jid]) != int(mujoco.mjtJoint.mjJNT_FREE):
            continue
        adr = int(model.jnt_qposadr[jid])
        # Keep XY at the rest pose; preserve relative vertical motion around rest Z.
        rest_xyz = np.asarray(model.qpos0[adr : adr + 3], dtype=np.float64)
        out[:, adr] = rest_xyz[0]
        out[:, adr + 1] = rest_xyz[1]
        out[:, adr + 2] = rest_xyz[2] + (out[:, adr + 2] - out[0, adr + 2])
    return out


def compose_timeline(
    model: mujoco.MjModel,
    cache: Path,
    clips: tuple[str, ...],
    seconds_per_clip: float,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Return (qpos[T,nq], skin_alpha[T], labels[T])."""
    q_blocks: list[np.ndarray] = []
    labels: list[str] = []
    lengths: list[int] = []
    for rel in clips:
        seg = load_clip_qpos(model, _resolve_clip(rel, cache))
        q = subsample(seg.qpos, seg.frequency_hz, FPS, seconds_per_clip)
        q = pin_root_xy(model, q)
        q_blocks.append(q)
        lengths.append(len(q))
        labels.extend([seg.label] * len(q))
    qpos = np.concatenate(q_blocks, axis=0)
    alpha = compose_skin_alpha(lengths)
    return qpos, alpha, labels


def _fixed_lookat(model: mujoco.MjModel, data: mujoco.MjData) -> np.ndarray:
    """Subject center at the current pose — used once after pinning to lock the camera."""
    mins = np.full(3, np.inf)
    maxs = np.full(3, -np.inf)
    for gid in range(model.ngeom):
        if int(model.geom_bodyid[gid]) == 0:
            continue
        pos = data.geom_xpos[gid]
        pad = float(np.max(model.geom_size[gid])) + 0.05
        mins = np.minimum(mins, pos - pad)
        maxs = np.maximum(maxs, pos + pad)
    if not np.all(np.isfinite(mins)):
        return model.stat.center.copy()
    return 0.5 * (mins + maxs)


def _camera_for_frame(
    frame_idx: int,
    n_frames: int,
    lookat: np.ndarray,
) -> mujoco.MjvCamera:
    t = frame_idx / max(1, n_frames - 1)
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = lookat
    cam.distance = CAM_DISTANCE
    cam.azimuth = CAM_AZIMUTH_START + CAM_AZIMUTH_SPAN * t
    cam.elevation = CAM_ELEVATION
    return cam


def _scene_option(skin_alpha: float) -> mujoco.MjvOption:
    opt = mujoco.MjvOption()
    opt.sitegroup[:] = False
    # Hide wraps (3) and collision (4). Cross-dissolve bones with the skin fade.
    show_bones = skin_alpha < 0.92
    for g in range(6):
        if g in (3, 4):
            opt.geomgroup[g] = False
        elif g == 0:
            opt.geomgroup[g] = show_bones
        else:
            opt.geomgroup[g] = True
    # Tendons poke through the HM08 surface — ease them out with the skin fade.
    opt.flags[mujoco.mjtVisFlag.mjVIS_TENDON] = skin_alpha < 0.45
    return opt


def _write_mp4_ffmpeg(frames_iter, out_path: Path, fps: int) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-y",
        "-f",
        "rawvideo",
        "-vcodec",
        "rawvideo",
        "-s",
        f"{WIDTH}x{HEIGHT}",
        "-pix_fmt",
        "rgb24",
        "-r",
        str(fps),
        "-i",
        "-",
        "-an",
        "-vcodec",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-crf",
        "18",
        str(out_path),
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    assert proc.stdin is not None
    try:
        for frame in frames_iter:
            proc.stdin.write(np.asarray(frame, dtype=np.uint8).tobytes())
    finally:
        proc.stdin.close()
        stderr = proc.stderr.read().decode("utf-8", errors="replace") if proc.stderr else ""
        code = proc.wait()
    if code != 0:
        raise RuntimeError(f"ffmpeg failed ({code}):\n{stderr[-2000:]}")


def render_teaser(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    qpos: np.ndarray,
    alphas: np.ndarray,
    out_path: Path,
) -> None:
    n = len(qpos)
    print(f"Rendering {n} frames ({n / FPS:.1f}s) → {out_path}")

    # Lock look-at once (root XY is pinned, so no spatial pan).
    data.qpos[:] = qpos[0]
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)
    lookat = _fixed_lookat(model, data)

    def frames():
        with mujoco.Renderer(model, height=HEIGHT, width=WIDTH) as renderer:
            for i in range(n):
                data.qpos[:] = qpos[i]
                data.qvel[:] = 0.0
                mujoco.mj_forward(model, data)
                a = float(alphas[i])
                model.skin_rgba[0, 3] = a
                cam = _camera_for_frame(i, n, lookat)
                opt = _scene_option(a)
                renderer.update_scene(data, camera=cam, scene_option=opt)
                yield renderer.render()
                if i % 30 == 0:
                    print(f"  frame {i}/{n}  skin_alpha={a:.2f}")

    _write_mp4_ffmpeg(frames(), out_path, FPS)
    print(f"Wrote {out_path}")


def run_mjviser(model: mujoco.MjModel, qpos: np.ndarray, alphas: np.ndarray) -> None:
    """Interactive scrubber for the teaser trajectory (geom view; no MuJoCo skins)."""
    try:
        import viser
        from mjviser import ViserMujocoScene
    except ImportError as exc:
        raise SystemExit(
            "mjviser is not installed in this environment.\n"
            "Install the local checkout, e.g.\n"
            "  uv pip install -e /Users/caggiano/Dropbox/Myo/mjviser\n"
            f"Original error: {exc}"
        ) from exc

    print(
        "mjviser note: MuJoCo <skin> meshes are not drawn by mjviser; "
        "use the MP4 teaser for skin in/out. This view scrubs the same motion."
    )
    server = viser.ViserServer()
    scene = ViserMujocoScene(server, model, num_envs=1)
    data = mujoco.MjData(model)
    n = len(qpos)
    frame_idx = [0]
    playing = [True]
    tabs = scene.create_visualization_gui()
    with tabs.add_tab("Teaser", icon=viser.Icon.PLAYER_PLAY):
        timeline = server.gui.add_slider("Frame", min=0, max=n - 1, step=1, initial_value=0)
        alpha_label = server.gui.add_html("")
        play_btn = server.gui.add_button("Pause", icon=viser.Icon.PLAYER_PAUSE)

        @play_btn.on_click
        def _(_) -> None:
            playing[0] = not playing[0]
            play_btn.label = "Pause" if playing[0] else "Play"

        @timeline.on_update
        def _(_) -> None:
            frame_idx[0] = int(timeline.value)

        def show(i: int) -> None:
            data.qpos[:] = qpos[i]
            data.qvel[:] = 0.0
            mujoco.mj_forward(model, data)
            scene.update_from_mjdata(data)
            alpha_label.content = (
                f"<div style='font-family:monospace'>t={i / FPS:.2f}s  skin_alpha={alphas[i]:.2f} (MP4 only)</div>"
            )

        show(0)
        import time

        last = time.perf_counter()
        acc = 0.0
        try:
            while True:
                now = time.perf_counter()
                dt = now - last
                last = now
                if playing[0]:
                    acc += dt
                    while acc >= 1.0 / FPS:
                        acc -= 1.0 / FPS
                        frame_idx[0] = (frame_idx[0] + 1) % n
                        timeline.value = frame_idx[0]
                        show(frame_idx[0])
                time.sleep(1 / 60)
        except KeyboardInterrupt:
            print("\nStopped.")
            server.stop()


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    p.add_argument("--out", type=Path, default=OUT_DIR / "hm08_skin_teaser.mp4")
    p.add_argument("--seconds-per-clip", type=float, default=5.0)
    p.add_argument("--mjviser", action="store_true", help="Interactive scrubber instead of MP4")
    p.add_argument("--clips", nargs="*", default=list(DEFAULT_CLIPS))
    args = p.parse_args(argv)

    model, data = build_model_with_skin()
    qpos, alphas, labels = compose_timeline(
        model,
        args.cache,
        tuple(args.clips),
        args.seconds_per_clip,
    )
    print("Clips:", ", ".join(dict.fromkeys(labels)))
    print(f"Total duration: {len(qpos) / FPS:.1f}s  nskin={model.nskin}")

    if args.mjviser:
        run_mjviser(model, qpos, alphas)
        return 0

    render_teaser(model, data, qpos, alphas, args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
