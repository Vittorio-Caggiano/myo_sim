# Wiki Log

Append-only maintenance log, most recent first.

## 2026-10-31 — Arm rigid-tendon maps from MoBL-ARMS, checked along measured shoulder motion
Changed: `myo_sim/models/arm/assets/myoarm_r_muscle.xml`, `myo_sim/build/rigid_conversion.py` (regions), `myo_sim/build/data/mobl_arm_muscles.json`, `tests/test_arm_rigid_conversion.py`, `tests/data/seth2019_shoulder_motion.npz`, `myo_sim/models/arm/README.md`
Why: Along the measured shoulder motions of Seth et al. (2019), LAT1, LAT3 and PECM3 operated outside 0.5-1.5 L0 in 100 %, 64 % and 56 % of frames. Ten muscles now use maps converted directly from MoBL-ARMS (0-5 % of frames outside, closer to the Seth model's fibre lengths); the selection rules are recorded in the data file.

## 2026-10-31 — Leg passive force calibrated to measured passive joint moments
Changed: `myo_sim/models/leg/assets/myolegs_muscle.xml`, `myo_sim/build/leg_passive.py`, `myo_sim/build/data/leg_passive.json`, `tests/test_leg_passive.py`, `tests/test_leg_rigid_conversion.py`, `tests/data/rajagopal2016_leg_opensim.npz`, `tests/debug_leg_physiology.py`, `myo_sim/models/leg/README.md`
Why: With the converted maps the hamstrings still gave 19-41 Nm of passive knee moment where people show 6-30 Nm, and MuJoCo's default passive scale did not reflect measured passive moments anywhere. Tendon slack length and fpmax per muscle are fitted to Silder et al. (2007), keeping the gastrocnemius at Narici et al. (1996) fascicle lengths and each muscle at half its active force or more at Anderson et al. (2007) strength angles; hip/knee/ankle RMS 19.8/11.2/2.6 -> 11.0/8.4/2.1 N m. Torques in the physiology script now use effective moment arms (knee translations and patella resolved), which changes the knee numbers reported on this date.

## 2026-10-31 — Leg rigid-tendon muscle maps derived directly from the Rajagopal reference
Changed: `myo_sim/models/leg/assets/myolegs_muscle.xml`, `myo_sim/build/rigid_conversion.py`, `myo_sim/build/data/rajagopal2016_leg_muscles.json`, `tests/test_leg_rigid_conversion.py`, `tests/debug_leg_physiology.py`, `docs/wiki/model-authoring.md`, `docs/wiki/testing-guide.md`, `docs/wiki/repository-map.md`, `myo_sim/models/leg/README.md`
Why: The 2026-08-02 leg refit kept anatomical optimal fibre lengths but anchored them to a gait-range `lengthrange`; outside it MuJoCo extrapolates, and semimembranosus reached ln ~2.9 with the hip flexed (161 Nm passive knee moment). Fitting MuJoCo's linear map to the reference's rigid-tendon fibre length over the full joint range reproduces OpenSim's fibre lengths (median RMSE 0.02, max 0.10, over 400 postures) and was adopted for the 24 muscles where it raises the isometric force error by at most 2 % of Fmax; checked against cadaver fibre lengths, ultrasound fascicle lengths and dynamometry.

## 2026-10-27 — Head and trunk mass convention
Changed: myo_sim/models/head/assets/myohead_rigid_chain.xml, myo_sim/models/torso/assets/myotorso_chain.xml, myo_sim/build/mass.py, tests/test_mass_convention.py, CHANGELOG.md
Why: the OpenSim torso mass already contained the head, leaving the head at about 2.5 % of body mass; head set to 6.94 % (de Leva) at unchanged total mass.
