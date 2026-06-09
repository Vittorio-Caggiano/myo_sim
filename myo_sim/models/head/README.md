# MyoHead

Rigid kinematic head scaffold derived from the OpenSim HAT (Head-Arms-Trunk) body segment, used as an attachment point in full-body compositions.

## Anatomical scope

| Property | Value |
|---|---|
| Degrees of freedom | 0 (fully rigid — neck joints commented out) |
| Actuators (muscles) | 0 |
| Body segments | neck, head |
| Primary joints | none active (neck_rotation and neck_flexion defined but disabled) |

## Reference model

- **Source:** <!-- TODO: review — HAT segment origin inferred from mesh prefix `hat_`; confirm source -->
- **Paper:** <!-- TODO: review -->

## Fidelity

<!-- TODO: review -->

## Known limitations

None currently tracked.

## Manual adjustments

- Both neck joints (`neck_rotation`, `neck_flexion`) are commented out to produce a fully rigid head for use in `myofullbody` composition. The joint definitions remain in the file for reference.
- Collision geometry added by the MuscleMimic team (2025–2026): capsule for cervical region, two ellipsoids for jaw and skull.
- Entry point updated from `myohead_simple_chain.xml` to `myohead_rigid_chain.xml` to reflect the fully locked configuration.

## Changelog

**2026-06-10** — Fixed `myohead_simple.xml` to include `myohead_rigid_chain.xml` instead of the now-removed `myohead_simple_chain.xml`.

**2026-06-03** — Refactored model structure; head added as rigid scaffold for full-body assembly.

## Citation

See repository README.
