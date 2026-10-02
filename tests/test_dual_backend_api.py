"""Dual-backend API: C plugin and Warp share sprinter assets; short rollouts stay finite."""

from __future__ import annotations

import os

import numpy as np
import pytest

pytest.importorskip("mujoco")


def _plugin_or_skip():
    from myo_sim.plugin import ensure_built, load_plugin

    try:
        ensure_built()
    except RuntimeError as e:
        pytest.skip(str(e))
    load_plugin()


def _warp_or_skip():
    pytest.importorskip("warp")
    pytest.importorskip("mujoco_warp")
    import warp as wp

    wp.init()
    os.environ.setdefault("VV_WARP_DEVICE", "cpu")


def _nbody_from_xml(xml_path) -> int:
    import mujoco

    return mujoco.MjModel.from_xml_path(str(xml_path)).nbody


def test_make_musculoskeletal_c_rollout(tmp_path):
    _plugin_or_skip()
    from myo_sim import sprinter_assets
    from myo_sim.physics import make_musculoskeletal

    muscles = sprinter_assets.load_muscles()
    xml = sprinter_assets.xml_path()
    nbody = _nbody_from_xml(xml)

    # Muscle-only smoke (no stiff HC contacts / coordinate limits) so a short free-fall stays finite.
    drv = make_musculoskeletal("c", muscles, contact=None, nworld=1, limits=(), nbody=nbody)
    m, d = drv.prepare_model(xml, spec_path=tmp_path / "c.spec", out_xml=tmp_path / "c_plugin.xml")
    drv.equilibrate()
    q0 = d.qpos.copy()
    d.ctrl[:] = 0.0
    for _ in range(10):
        drv.step()
    assert np.all(np.isfinite(d.qpos))
    assert np.linalg.norm(d.qpos - q0) > 0
    st = drv.muscle.state
    assert st.shape == (1, len(muscles), 10)
    assert np.all(np.isfinite(st))


def test_make_musculoskeletal_warp_rollout():
    _warp_or_skip()
    from myo_sim import sprinter_assets
    from myo_sim.physics import make_musculoskeletal

    muscles = sprinter_assets.load_muscles()
    xml = sprinter_assets.xml_path()
    nbody = _nbody_from_xml(xml)

    # Muscle-only (no HC contacts): same as C smoke — stiff contacts + mjw.step are chaotic/NaN-prone.
    drv = make_musculoskeletal("warp", muscles, contact=None, nworld=1, limits=(), nbody=nbody, device="cpu")
    m, d = drv.prepare_model(xml)
    drv.equilibrate()
    q0 = d.qpos.numpy().copy()
    for _ in range(10):
        ctrl = np.zeros((1, drv.mjm.nu), dtype=np.float32)
        d.ctrl.assign(ctrl)
        drv.step()
    q1 = d.qpos.numpy()
    assert np.all(np.isfinite(q1))
    assert np.linalg.norm(q1 - q0) > 0


def test_c_vs_warp_muscle_state_and_qfrc(tmp_path):
    """Teacher-forced: identical qpos/qvel/ctrl/act/ln → muscle force and muscle qfrc within f32 envelope.

    Does not free-roll with Hunt–Crossley (chaotic under contact).
    """
    _plugin_or_skip()
    _warp_or_skip()
    import mujoco
    import mujoco_warp as mjw

    from myo_sim import sprinter_assets
    from myo_sim.plugin import cmyo
    from myo_sim.warp.muscle_system import MuscleSystem as WMuscle

    muscles = sprinter_assets.load_muscles()
    limits = sprinter_assets.load_limits()
    xml = sprinter_assets.xml_path()
    n = len(muscles)
    pv = np.stack([np.asarray(m["p"], dtype=np.float64) for m in muscles])

    spec = tmp_path / "cmp.spec"
    cmyo.write_spec(spec, muscles, limits=limits, contact=None)
    out = tmp_path / "cmp_plugin.xml"
    out.write_text(cmyo.inject_plugin_xml(xml.read_text(), str(spec.resolve())))
    m_c = mujoco.MjModel.from_xml_path(str(out))
    d_c = mujoco.MjData(m_c)
    m_np = mujoco.MjModel.from_xml_path(str(xml))
    d_np = mujoco.MjData(m_np)
    wm = mjw.put_model(m_np)
    wd = mjw.put_data(m_np, mujoco.MjData(m_np), nworld=1)
    wmus = WMuscle(muscles, 1, limits=limits, device="cpu")

    rng = np.random.default_rng(11)
    qpos = m_c.qpos0.copy()
    qpos[7:] += rng.normal(0, 0.02, m_c.nq - 7)
    qvel = rng.normal(0, 0.05, m_c.nv)
    ctrl = rng.uniform(0.1, 0.9, m_c.nu)

    d_c.qpos[:] = qpos
    d_c.qvel[:] = qvel
    d_c.ctrl[:] = ctrl
    mujoco.mj_forward(m_c, d_c)
    st = cmyo.state_view(m_c, d_c, n)
    st[:, 1] = cmyo.equilibrium(st[:, 3].copy(), st[:, 4].copy(), st[:, 0].copy(), pv)
    mujoco.mj_forward(m_c, d_c)
    st_c = cmyo.state_view(m_c, d_c, n).copy()
    d_np.qpos[:] = qpos
    d_np.qvel[:] = qvel
    d_np.ctrl[:] = ctrl
    mujoco.mj_forward(m_np, d_np)
    qfrc_c = d_c.qfrc_passive - d_np.qfrc_passive

    wd.qpos.assign(qpos.astype(np.float32)[None])
    wd.qvel.assign(qvel.astype(np.float32)[None])
    wd.ctrl.assign(ctrl.astype(np.float32)[None])
    mjw.fwd_position(wm, wd)
    mjw.fwd_velocity(wm, wd)
    buf = wmus.state.numpy()
    buf[0, :, 0] = st_c[:, 0].astype(np.float32)
    buf[0, :, 1] = st_c[:, 1].astype(np.float32)
    wmus.state.assign(buf)
    wmus.compute(wd)
    wmus.apply(wd, wm, zero=True)
    st_w = wmus.state.numpy()[0]
    qfrc_w = wd.qfrc_applied.numpy()[0]

    dF = float(np.max(np.abs(st_c[:, 2] - st_w[:, 2])))
    dL = float(np.max(np.abs(st_c[:, 3] - st_w[:, 3])))
    dq = float(np.max(np.abs(qfrc_c - qfrc_w)))
    assert dL <= 1e-5, dL
    assert dF <= 1.0 or dF / max(float(np.max(np.abs(st_c[:, 2]))), 1.0) <= 1e-4
    assert dq <= 5.0 or dq / max(float(np.max(np.abs(qfrc_w))), 1.0) <= 1e-3
