import mujoco
import pytest

from myo_sim.build.compose import apply_site_pos_overrides, build_spec


def test_site_pos_overrides_apply_and_compile():
    spec = build_spec("myotorso")
    site_names = [site.name for site in spec.sites][:2]
    new_positions = {name: (1.0, 2.0, 3.0) for name in site_names}

    apply_site_pos_overrides(spec, new_positions)

    for name, pos in new_positions.items():
        assert tuple(spec.site(name).pos) == pos

    model = spec.compile()

    assert isinstance(model, mujoco.MjModel)


def test_site_pos_overrides_only_touches_named_sites():
    spec = build_spec("myotorso")
    target_name = spec.sites[0].name
    untouched_name = spec.sites[1].name
    untouched_pos_before = tuple(spec.site(untouched_name).pos)

    apply_site_pos_overrides(spec, {target_name: (1.0, 2.0, 3.0)})

    assert tuple(spec.site(untouched_name).pos) == untouched_pos_before


def test_site_pos_overrides_none_is_a_no_op():
    baseline = build_spec("myotorso").compile()
    overridden = build_spec("myotorso")
    apply_site_pos_overrides(overridden, {})
    overridden = overridden.compile()

    assert overridden.nbody == baseline.nbody
    assert (overridden.site_pos == baseline.site_pos).all()


def test_unknown_site_name_raises_keyerror():
    spec = build_spec("myotorso")

    with pytest.raises(KeyError, match="not_a_real_site"):
        apply_site_pos_overrides(spec, {"not_a_real_site": (0.0, 0.0, 0.0)})
