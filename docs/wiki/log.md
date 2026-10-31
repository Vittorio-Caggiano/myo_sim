# Wiki Log

Append-only maintenance log, most recent first.

## 2026-10-31 — Leg rigid-tendon muscle maps derived directly from the Rajagopal reference
Changed: `myo_sim/models/leg/assets/myolegs_muscle.xml`, `myo_sim/build/rigid_conversion.py`, `myo_sim/build/data/rajagopal2016_leg_muscles.json`, `tests/test_leg_rigid_conversion.py`, `tests/debug_leg_physiology.py`, `docs/wiki/model-authoring.md`, `docs/wiki/testing-guide.md`, `docs/wiki/repository-map.md`, `myo_sim/models/leg/README.md`
Why: The 2026-08-02 leg refit kept anatomical optimal fibre lengths but anchored them to a gait-range `lengthrange`; outside it MuJoCo extrapolates, and semimembranosus reached ln ~2.9 with the hip flexed (138 Nm passive knee moment). Fitting MuJoCo's linear map to the reference's rigid-tendon fibre length over the full joint range reproduces OpenSim's fibre lengths (median RMSE 0.02, max 0.10, over 400 postures) and was adopted for the 24 muscles where it also does not raise the isometric force error; checked against cadaver fibre lengths, ultrasound fascicle lengths and dynamometry.

## 2026-10-27 — Head and trunk mass convention
Changed: myo_sim/models/head/assets/myohead_rigid_chain.xml, myo_sim/models/torso/assets/myotorso_chain.xml, myo_sim/build/mass.py, tests/test_mass_convention.py, CHANGELOG.md
Why: the OpenSim torso mass already contained the head, leaving the head at about 2.5 % of body mass; head set to 6.94 % (de Leva) at unchanged total mass.
