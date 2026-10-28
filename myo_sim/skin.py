"""Procedural MuJoCo skin (``.skn``) attach for MyoSim models.

The packaged HM08 full-body skin can be attached to ``myofullbody`` as-is, or
filtered to a body-segment subset (arm / legs / torso) so the same asset works
on fragment models that omit missing bind bodies.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Iterable, Sequence

import numpy as np

from myo_sim import MODELS_DIR

if TYPE_CHECKING:
    import mujoco

HM08_SKN = MODELS_DIR / "skins" / "hm08" / "myofullbody.skn"

# Named segment filters over bind-body names stored in the HM08 SKN.
# Intersection with bodies present on the target MjSpec is applied at attach time.
SEGMENTS: dict[str, frozenset[str]] = {
    "arm_r": frozenset(
        {
            "clavicle_r",
            "scapula_r",
            "humerus_r",
            "ulna_r",
            "radius_r",
            "lunate_r",
            "firstmc_r",
            "proximal_thumb_r",
            "distal_thumb_r",
            "2proxph_r",
            "midph2_r",
            "distph2_r",
            "3proxph_r",
            "midph3_r",
            "distph3_r",
            "4proxph_r",
            "midph4_r",
            "distph4_r",
            "5proxph_r",
            "midph5_r",
            "distph5_r",
        }
    ),
    "arm_l": frozenset(
        {
            "clavicle_l",
            "scapula_l",
            "humerus_l",
            "ulna_l",
            "radius_l",
            "lunate_l",
            "firstmc_l",
            "proximal_thumb_l",
            "distal_thumb_l",
            "2proxph_l",
            "midph2_l",
            "distph2_l",
            "3proxph_l",
            "midph3_l",
            "distph3_l",
            "4proxph_l",
            "midph4_l",
            "distph4_l",
            "5proxph_l",
            "midph5_l",
            "distph5_l",
        }
    ),
    "legs": frozenset(
        {
            "pelvis",
            "femur_r",
            "tibia_r",
            "calcn_r",
            "toes_r",
            "femur_l",
            "tibia_l",
            "calcn_l",
            "toes_l",
        }
    ),
    "torso": frozenset(
        {
            "pelvis",
            "lumbar5",
            "lumbar4",
            "lumbar3",
            "lumbar2",
            "lumbar1",
            "torso",
            "head",
            "clavicle_r",
            "clavicle_l",
            "scapula_r",
            "scapula_l",
        }
    ),
}


@dataclass(frozen=True)
class SkinBone:
    body: str
    bindpos: np.ndarray
    bindquat: np.ndarray
    vertid: np.ndarray
    vertweight: np.ndarray


@dataclass(frozen=True)
class SkinMesh:
    """Parsed MuJoCo binary skin."""

    vertices: np.ndarray
    faces: np.ndarray
    bones: tuple[SkinBone, ...]
    texcoord: np.ndarray | None = None


def read_skn(path: Path | str) -> SkinMesh:
    """Parse a MuJoCo ``.skn`` file."""
    raw = Path(path).read_bytes()
    nvert, ntexcoord, nface, nbones = struct.unpack_from("<4i", raw)
    off = 16
    vertices = np.frombuffer(raw, dtype="<f4", count=nvert * 3, offset=off).reshape(nvert, 3).copy()
    off += nvert * 12
    texcoord = None
    if ntexcoord:
        texcoord = np.frombuffer(raw, dtype="<f4", count=ntexcoord * 2, offset=off).reshape(ntexcoord, 2).copy()
        off += ntexcoord * 8
    faces = np.frombuffer(raw, dtype="<i4", count=nface * 3, offset=off).reshape(nface, 3).copy()
    off += nface * 12
    bones: list[SkinBone] = []
    for _ in range(nbones):
        body = raw[off : off + 40].split(b"\0", 1)[0].decode("ascii")
        off += 40
        bindpos = np.frombuffer(raw, dtype="<f4", count=3, offset=off).copy()
        off += 12
        bindquat = np.frombuffer(raw, dtype="<f4", count=4, offset=off).copy()
        off += 16
        (nbind,) = struct.unpack_from("<i", raw, off)
        off += 4
        vertid = np.frombuffer(raw, dtype="<i4", count=nbind, offset=off).copy()
        off += nbind * 4
        vertweight = np.frombuffer(raw, dtype="<f4", count=nbind, offset=off).copy()
        off += nbind * 4
        bones.append(SkinBone(body, bindpos, bindquat, vertid, vertweight))
    if off != len(raw):
        raise ValueError(f"SKN parse left {len(raw) - off} unread bytes in {path}")
    return SkinMesh(vertices=vertices, faces=faces, bones=tuple(bones), texcoord=texcoord)


def write_skn(path: Path | str, skin: SkinMesh) -> None:
    """Write a MuJoCo ``.skn`` file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    nvert = int(skin.vertices.shape[0])
    nface = int(skin.faces.shape[0])
    tex = skin.texcoord
    ntexcoord = nvert if tex is not None else 0
    if tex is not None and tex.shape[0] != nvert:
        raise ValueError("texcoord count must match vertex count")
    with path.open("wb") as out:
        out.write(struct.pack("<4i", nvert, ntexcoord, nface, len(skin.bones)))
        out.write(np.asarray(skin.vertices, dtype="<f4").tobytes())
        if tex is not None:
            out.write(np.asarray(tex, dtype="<f4").tobytes())
        out.write(np.asarray(skin.faces, dtype="<i4").tobytes())
        for bone in skin.bones:
            name = bone.body.encode("ascii")
            if len(name) > 40:
                raise ValueError(f"Bone name too long for SKN: {bone.body}")
            out.write(name.ljust(40, b"\0"))
            out.write(np.asarray(bone.bindpos, dtype="<f4").tobytes())
            out.write(np.asarray(bone.bindquat, dtype="<f4").tobytes())
            out.write(struct.pack("<i", int(bone.vertid.size)))
            out.write(np.asarray(bone.vertid, dtype="<i4").tobytes())
            out.write(np.asarray(bone.vertweight, dtype="<f4").tobytes())


def filter_skin(
    skin: SkinMesh,
    bodies: Iterable[str],
    *,
    weight_floor: float = 1e-6,
) -> SkinMesh:
    """Keep only bind bones in ``bodies`` and the vertices/faces they influence."""
    keep_names = set(bodies)
    keep_bones = [b for b in skin.bones if b.body in keep_names]
    if not keep_bones:
        raise ValueError("No SKN bones remain after body filter")

    mass = np.zeros(skin.vertices.shape[0], dtype=np.float64)
    for bone in keep_bones:
        mass[bone.vertid] += bone.vertweight
    vkeep = np.flatnonzero(mass > weight_floor)
    if vkeep.size == 0:
        raise ValueError("No SKN vertices remain after body filter")

    remap = np.full(skin.vertices.shape[0], -1, dtype=np.int32)
    remap[vkeep] = np.arange(vkeep.size, dtype=np.int32)
    face_mask = np.all(remap[skin.faces] >= 0, axis=1)
    faces = remap[skin.faces[face_mask]]
    if faces.size == 0:
        raise ValueError("No SKN faces remain after body filter")

    bones: list[SkinBone] = []
    for bone in keep_bones:
        mapped = remap[bone.vertid]
        ok = mapped >= 0
        if not np.any(ok):
            continue
        bones.append(
            SkinBone(
                body=bone.body,
                bindpos=bone.bindpos.copy(),
                bindquat=bone.bindquat.copy(),
                vertid=mapped[ok],
                vertweight=bone.vertweight[ok].astype(np.float32, copy=True),
            )
        )
    if not bones:
        raise ValueError("No SKN bones with remaining vertex binds")

    tex = None if skin.texcoord is None else skin.texcoord[vkeep].copy()
    return SkinMesh(
        vertices=skin.vertices[vkeep].copy(),
        faces=faces.astype(np.int32, copy=False),
        bones=tuple(bones),
        texcoord=tex,
    )


def resolve_body_filter(
    *,
    segment: str | None = None,
    bodies: Sequence[str] | None = None,
    present: Iterable[str] | None = None,
) -> frozenset[str] | None:
    """Combine a named segment and/or explicit body list with model-present bodies.

    Returns ``None`` when no filtering is requested (attach the full SKN file).
    """
    selected: set[str] | None = None
    if segment is not None:
        if segment == "fullbody":
            selected = None
        elif segment not in SEGMENTS:
            raise ValueError(f"Unknown segment {segment!r}. Available: {sorted(SEGMENTS) + ['fullbody']}")
        else:
            selected = set(SEGMENTS[segment])
    if bodies is not None:
        extra = set(bodies)
        selected = extra if selected is None else selected & extra
    if present is not None:
        present_set = set(present)
        if selected is None:
            # Restrict to bodies that exist so fragment models can take the full asset.
            selected = present_set
        else:
            selected &= present_set
    return frozenset(selected) if selected is not None else None


def attach_skin(
    spec: "mujoco.MjSpec",
    skn_path: Path | str | None = None,
    *,
    name: str = "myofullbody_skin",
    segment: str | None = None,
    bodies: Sequence[str] | None = None,
    rgba: Sequence[float] = (0.78, 0.62, 0.52, 1.0),
    group: int = 2,
    material: str | None = None,
    filtered_skn_path: Path | str | None = None,
) -> "mujoco.MjsSkin":
    """Attach a ``.skn`` to ``spec``, optionally filtered to a body segment.

    For the unfiltered full-body case (every bind body present) the SKN is
    referenced by file. Otherwise a pruned skin is attached — written to
    ``filtered_skn_path`` when provided, else inlined on the spec.
    """
    path = Path(skn_path) if skn_path is not None else HM08_SKN
    if not path.is_file():
        raise FileNotFoundError(path)

    present = {body.name for body in spec.bodies}
    parsed = read_skn(path)
    skn_bodies = {bone.body for bone in parsed.bones}

    # Explicit segment/body filters always prune. Otherwise prune only when the
    # model is missing bind bodies (fragment models).
    wants_filter = segment not in (None, "fullbody") or bodies is not None
    if not wants_filter and skn_bodies.issubset(present):
        return spec.add_skin(
            name=name,
            file=str(path.resolve()),
            rgba=list(rgba),
            group=group,
            material=material,
        )

    body_filter = resolve_body_filter(segment=segment, bodies=bodies, present=present)
    if body_filter is None:
        body_filter = frozenset(present)
    skin = filter_skin(parsed, body_filter)

    if filtered_skn_path is not None:
        out = Path(filtered_skn_path)
        write_skn(out, skin)
        return spec.add_skin(
            name=name,
            file=str(out.resolve()),
            rgba=list(rgba),
            group=group,
            material=material,
        )

    return spec.add_skin(
        name=name,
        rgba=list(rgba),
        group=group,
        material=material,
        vert=skin.vertices.astype(np.float64).ravel().tolist(),
        texcoord=None if skin.texcoord is None else skin.texcoord.astype(np.float64).ravel().tolist(),
        face=skin.faces.astype(np.int32).ravel().tolist(),
        bodyname=[b.body for b in skin.bones],
        bindpos=np.concatenate([b.bindpos for b in skin.bones]).astype(np.float64).tolist(),
        bindquat=np.concatenate([b.bindquat for b in skin.bones]).astype(np.float64).tolist(),
        vertid=[b.vertid.astype(np.int32).tolist() for b in skin.bones],
        vertweight=[b.vertweight.astype(np.float64).tolist() for b in skin.bones],
    )


def load_model_with_skin(
    model_name: str,
    *,
    skn_path: Path | str | None = None,
    segment: str | None = None,
    bodies: Sequence[str] | None = None,
    skin_name: str = "myofullbody_skin",
    rgba: Sequence[float] = (0.78, 0.62, 0.52, 1.0),
    group: int = 2,
) -> tuple["mujoco.MjModel", "mujoco.MjData"]:
    """Build ``model_name``, attach the HM08 skin (or segment), and compile."""
    import mujoco

    from myo_sim import load_spec

    spec = load_spec(model_name)
    attach_skin(
        spec,
        skn_path,
        name=skin_name,
        segment=segment,
        bodies=bodies,
        rgba=rgba,
        group=group,
    )
    model = spec.compile()
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    return model, data
