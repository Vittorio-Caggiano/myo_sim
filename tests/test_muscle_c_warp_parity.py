"""Teacher-forced muscle parity: myo_sim.plugin (C float64) vs myo_sim.warp (float32).

Compares muscle state and muscle-only generalized forces at identical poses. C's
``qfrc_passive`` also includes joint damping; that baseline is subtracted using a
no-plugin forward so the residual matches Warp's ``qfrc_applied``.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("mujoco")


def _plugin_ready():
    from myo_sim.plugin import ensure_built, load_plugin

    try:
        ensure_built()
    except RuntimeError as e:
        pytest.skip(str(e))
    load_plugin()


def _warp_ready():
    pytest.importorskip("warp")
    pytest.importorskip("mujoco_warp")
    import warp as wp

    wp.init()
    os.environ.setdefault("VV_WARP_DEVICE", "cpu")


def test_muscle_c_vs_warp_teacher_forced():
    _plugin_ready()
    _warp_ready()

    import mujoco
    import mujoco_warp as mjw

    from myo_sim import sprinter_assets
    from myo_sim.plugin import cmyo
    from myo_sim.warp.muscle_system import MuscleSystem as WMuscle

    muscles = sprinter_assets.load_muscles()
    limits = sprinter_assets.load_limits()
    xml_path = sprinter_assets.xml_path()
    n = len(muscles)
    pv = np.stack([np.asarray(m["p"], dtype=np.float64) for m in muscles])

    tmp = Path(tempfile.mkdtemp(prefix="muscle_c_warp_"))
    spec = tmp / "muscles.spec"
    cmyo.write_spec(spec, muscles, limits=limits, contact=None)
    plugin_xml = tmp / "plugin.xml"
    plugin_xml.write_text(cmyo.inject_plugin_xml(xml_path.read_text(), str(spec.resolve())))

    m_c = mujoco.MjModel.from_xml_path(str(plugin_xml))
    d_c = mujoco.MjData(m_c)
    m_np = mujoco.MjModel.from_xml_path(str(xml_path))
    d_np = mujoco.MjData(m_np)

    wm = mjw.put_model(m_np)
    wd = mjw.put_data(m_np, mujoco.MjData(m_np), nworld=1)
    wmus = WMuscle(muscles, 1, limits=limits, device="cpu")

    rng = np.random.default_rng(7)
    dF = dL = dln = dadot = dq = 0.0
    relF = relq = 0.0
    for _ in range(24):
        qpos = m_c.qpos0.copy()
        qpos[7:] += rng.normal(0, 0.03, m_c.nq - 7)
        qvel = rng.normal(0, 0.1, m_c.nv)
        ctrl = rng.uniform(0.0, 1.0, m_c.nu)

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

        dF = max(dF, float(np.max(np.abs(st_c[:, 2] - st_w[:, 2]))))
        dL = max(dL, float(np.max(np.abs(st_c[:, 3] - st_w[:, 3]))))
        dln = max(dln, float(np.max(np.abs(st_c[:, 1] - st_w[:, 1]))))
        dadot = max(dadot, float(np.max(np.abs(st_c[:, 5] - st_w[:, 5]))))
        dq = max(dq, float(np.max(np.abs(qfrc_c - qfrc_w))))
        Fscale = max(float(np.max(np.abs(st_c[:, 2]))), 1.0)
        qscale = max(float(np.max(np.abs(qfrc_w))), 1.0)
        relF = max(relF, dF / Fscale)
        relq = max(relq, dq / qscale)

    # Float32 envelope for C (float64 plugin) vs Warp teacher-forced muscle state / qfrc.
    assert dL <= 1e-5, f"tendon length drift {dL}"
    assert dln <= 1e-5, f"fiber length drift {dln}"
    assert dF <= 1.0 or relF <= 1e-4, f"muscle force drift abs={dF} rel={relF}"
    assert dadot <= 1e-4, f"act_dot drift {dadot}"
    assert dq <= 5.0 or relq <= 1e-3, f"muscle qfrc drift abs={dq} rel={relq}"


def test_ctypes_contraction_matches_plugin_state():
    """Batch ctypes contraction agrees with plugin_state force after mj_forward."""
    _plugin_ready()
    import mujoco

    from myo_sim import sprinter_assets
    from myo_sim.plugin import cmyo

    muscles = sprinter_assets.load_muscles()
    n = len(muscles)
    pv = np.stack([np.asarray(m["p"], dtype=np.float64) for m in muscles])
    tmp = Path(tempfile.mkdtemp(prefix="ctypes_vs_plugin_"))
    spec = tmp / "m.spec"
    cmyo.write_spec(spec, muscles, limits=(), contact=None)
    out = tmp / "p.xml"
    out.write_text(cmyo.inject_plugin_xml(sprinter_assets.xml_path().read_text(), str(spec.resolve())))
    m = mujoco.MjModel.from_xml_path(str(out))
    d = mujoco.MjData(m)
    d.ctrl[:] = 0.4
    mujoco.mj_forward(m, d)
    st = cmyo.state_view(m, d, n)
    cout = cmyo.contraction(st[:, 3].copy(), st[:, 4].copy(), st[:, 0].copy(), st[:, 1].copy(), pv)
    assert np.max(np.abs(cout["tendon_force"] - st[:, 2])) == 0.0
