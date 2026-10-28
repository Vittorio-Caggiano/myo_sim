"""Reproduce/track myo_sim#52: torso muscles flipping flexor/extensor role.

Several myotorso abdominal-wall muscles (rectus abdominis, external/internal
obliques) are single-role muscles with respect to trunk flexion/extension --
e.g. rectus abdominis is a flexor throughout the physiological range of
motion, never an extensor. If a muscle's moment arm about ``flex_extension``
changes sign within the joint's range, the muscle's mechanical role flips as
the trunk bends.

Root-caused against the model's source: ``myotorso`` is converted from
SimTK project 210 ("Constrained Lumbar Spine model", ``Lumbar_C_238_os4.osim``
-- see ``myo_sim/models/torso/README.md``). Checking that source directly:

- ``rect_abd``'s flip was caused by myo_sim's own ``flex_extension`` joint
  range (``-77.3 deg`` to ``+43.2 deg``) exceeding the OpenSim source's
  declared range for the independent coordinate ``L5_S1_Flex_Ext``
  (``-70.0 deg`` to ``+26.0 deg``). Tightening the MuJoCo joint range to match
  the source fixed it -- verified fixed below.
- The 12 oblique fascicles (``EO1-6``, ``IO1-6``) still flip *within* that
  same OpenSim-valid range. Their ``GeometryPath`` in the source ``.osim``
  has an empty ``PathWrapSet`` -- no wrap object protects them there either
  -- so this is a defect inherited from the third-party source model, not
  introduced by the MuJoCo port.

  Verified rigorously against actual OpenSim forward kinematics (opensim
  4.6, ``Lumbar_C_238_os4.osim``): along the way, found and fixed an
  unrelated ~20mm systematic mis-placement of the ``Abdomen`` body (see
  ``myotorso_chain.xml``'s ``Abdomen`` body pos) that had been silently
  offsetting every oblique fascicle's origin/insertion at neutral pose.
  After that fix, the 12 fascicles' path points match OpenSim's own
  positions to sub-micrometer precision at every pose used for calibration
  -- and they *still* flip identically. That rules out a myo_sim
  conversion bug: the flip is a property of the source model's own
  digitized via-points swept across its own declared range, not something
  fixable by importing a "stable value" from source. There is none.
  Fixing this requires adding wrap geometry that doesn't exist upstream --
  new anatomical modeling, not a port correction. Tracked as an expected
  failure until that's done.

See https://github.com/MyoHub/myo_sim/issues/52.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import mujoco  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402
from muscle_analysis_utils import compute_moment_arm_curve, parse_model_joint_equalities  # noqa: E402

import myo_sim  # noqa: E402

# rect_abd's flip was fixed by tightening flex_extension's range to match
# the OpenSim source (see module docstring) -- kept here as a regression
# guard. The oblique fascicles flip within the source's own valid range and
# have no wrap protection upstream either; tracked as known-broken.
FIXED_SINGLE_ROLE_FLEX_EXTENSION_MUSCLES = ("rect_abd",)
KNOWN_BROKEN_SINGLE_ROLE_FLEX_EXTENSION_MUSCLES = (
    "EO1",
    "EO2",
    "EO3",
    "EO4",
    "EO5",
    "EO6",
    "IO1",
    "IO2",
    "IO3",
    "IO4",
    "IO5",
    "IO6",
)

# A moment arm that changes sign and whose smaller-magnitude side is at
# least this fraction of the larger side counts as a genuine role flip,
# not numerical noise near a near-zero crossing.
FLIP_RATIO_THRESHOLD = 0.05


def _moment_arm_flip_ratio(ma: np.ndarray) -> float:
    """Ratio of the smaller-magnitude sign's peak to the larger, or 0 if no flip."""
    pos_peak = ma[ma > 0].max(initial=0.0)
    neg_peak = -ma[ma < 0].min(initial=0.0)
    if pos_peak == 0.0 or neg_peak == 0.0:
        return 0.0
    return min(pos_peak, neg_peak) / max(pos_peak, neg_peak)


def _flipping_muscles(base_muscles):
    model, data = myo_sim.load("myotorso")
    eq_map = parse_model_joint_equalities(model)
    jnt_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "flex_extension")
    assert jnt_id >= 0

    failures = []
    for base_muscle in base_muscles:
        for suffix in ("_r", "_l"):
            muscle = base_muscle + suffix
            act_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, muscle)
            if act_id < 0:
                continue
            tendon_id = model.actuator_trnid[act_id, 0]
            if tendon_id < 0:
                continue

            _, ma = compute_moment_arm_curve(model, data, tendon_id, jnt_id, eq_map=eq_map)
            if ma is None:
                continue

            ratio = _moment_arm_flip_ratio(ma)
            if ratio > FLIP_RATIO_THRESHOLD:
                failures.append((muscle, float(ma.min()), float(ma.max()), ratio))
    return failures


def _format_failures(failures):
    return "Muscles with a sign-flipping flex_extension moment arm (muscle, min, max, flip_ratio):\n" + "\n".join(
        f"  {name}: min={lo:.4f} max={hi:.4f} ratio={ratio:.2f}" for name, lo, hi, ratio in failures
    )


def test_rect_abd_does_not_flip_flexor_extensor_role():
    """Regression guard: fixed by tightening flex_extension's range to the OpenSim source's bound."""
    failures = _flipping_muscles(FIXED_SINGLE_ROLE_FLEX_EXTENSION_MUSCLES)
    assert not failures, _format_failures(failures)


@pytest.mark.xfail(reason="Inherited from OpenSim source: EO/IO have no wrap protection there either (myo_sim#52)", strict=True)
def test_obliques_do_not_flip_flexor_extensor_role():
    failures = _flipping_muscles(KNOWN_BROKEN_SINGLE_ROLE_FLEX_EXTENSION_MUSCLES)
    assert not failures, _format_failures(failures)


# Sites fixed to the Abdomen body, verified against actual OpenSim forward
# kinematics (opensim 4.6, Lumbar_C_238_os4.osim) to match source to
# sub-micrometer precision once Abdomen's pos was corrected from
# "0.098 0.0711 0" to "0.09 0.09 0" -- a ~20mm systematic offset that had
# been silently shifting every one of these muscles' origin/insertion.
ABDOMEN_ATTACHED_SITES = (
    "EO1_EO1-P1_r",
    "EO2_EO2-P2_r",
    "EO3_EO3-P2_r",
    "EO4_EO4-P2_r",
    "IO1_IO1-P1_r",
    "IO2_IO2-P1_r",
    "IO3_IO3-P1_r",
)


def test_abdomen_body_position_matches_opensim_source():
    """Regression guard for the Abdomen body placement fix (see module docstring)."""
    import mujoco as mj

    model, data = myo_sim.load("myotorso")
    mj.mj_forward(model, data)

    abdomen_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, "Abdomen")
    assert abdomen_id >= 0
    np.testing.assert_allclose(model.body_pos[abdomen_id], [0.09, 0.09, 0.0], atol=1e-6)

    for site_name in ABDOMEN_ATTACHED_SITES:
        sid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_SITE, site_name)
        assert sid >= 0, site_name
        assert model.site_bodyid[sid] == abdomen_id, site_name
