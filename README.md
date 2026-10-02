# MyoSim

[![CI](https://github.com/MyoHub/myo_sim/actions/workflows/ci.yml/badge.svg?branch=dev)](https://github.com/MyoHub/myo_sim/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-%3E%3D3.10-blue.svg)](pyproject.toml)
[![PyPI](https://img.shields.io/pypi/v/myo-sim)](https://pypi.org/project/myo-sim/)

MyoSim is the MuJoCo musculoskeletal model library used by [MyoSuite](https://github.com/facebookresearch/myoSuite).
It provides anatomically detailed XML models of the human arm, leg, torso, and hand,
plus a Python package for loading and composing them.

## Models

| Model | DoF | Muscles | Preview | Version |
|---|---:|---:|---|---|
| [**MyoLeg**](myo_sim/models/leg/README.md) | 29 | 80 | <img width="160" alt="Screenshot 2026-08-07 at 9 00 43 AM" src="https://github.com/user-attachments/assets/687308a3-29f0-4dfd-853f-5765e242dd62" />| stable |
| [**MyoArm**](myo_sim/models/arm/README.md) | 38 | 63 | <img alt="myoarm_demo" src="https://github.com/user-attachments/assets/eab77a8b-18af-445c-826f-b050b2a6ed7d" width="160"/> | stable |
| [**MyoTorso**](myo_sim/models/torso/README.md) (MyoBack) | 18 | 210 | <img src="https://github.com/cherylwang20/myo_sim/blob/cec3ce211a516a8798ed2edf9486a0814a0965da/MyoBack.png?raw=true" width="160"> | stable |
| [**MyoHand**](myo_sim/models/arm/README.md) | 23 | 39 | <img src="https://user-images.githubusercontent.com/23240128/232323950-39552200-614b-4c73-aab5-8a78daa0f5f3.png" width="160"> | stable |
| [**MyoFullBody**](docs/wiki/build-and-composition.md) | 123 | 416 | <img src="https://github.com/user-attachments/assets/37976636-1952-48a6-83c4-506722db4c82" width="160"/> | stable |
| [**MyoLeg26**](myo_sim/models/leg/README.md) | 18 | 26 | <img width="160" alt="Screenshot 2026-08-07 at 8 56 17 AM" src="https://github.com/user-attachments/assets/26b733d5-e47f-4d70-b2cc-b9e6f0fd1e7c" /> |beta |
| [**MyoFinger**](myo_sim/models/legacy/README.md) | 4 | 5 | <img src="https://user-images.githubusercontent.com/23240128/232323930-d1721f87-731b-432d-bafd-8c818ab4bbfe.png" width="160"> | legacy |
| [**MyoElbow**](myo_sim/models/legacy/README.md) | 2 | 6 | <img src="https://user-images.githubusercontent.com/23240128/232323890-6a601a82-1d3c-4e12-901c-0fd9cf232691.png" width="160"> | legacy |

Legacy models ship in the pip package under `myo_sim/models/legacy/` and load via registry names such as `myoelbow`, `myofinger`, and `osl`. They are maintained for backwards compatibility only — see `myo_sim/models/legacy/README.md`.

## Buildable Models

The following registered models can be compiled with `myo_sim.build.compose.build_model("<name>")`:

- `myotorso` — torso scaffold with torso muscles.
- `myotorso_abdomen` — simple abdomen scaffold.
- `myotorso_arm_r` — torso with the right arm only.
- `myotorso_arms` — torso with the right arm plus a mirrored-left arm.
- `myoarm_r` — passive anatomical torso scaffold with the right arm.
- `myoarms` — passive anatomical torso scaffold with mirrored arms.
- `myohand_r` — passive anatomical torso scaffold with the right hand derived from the pruned right arm.
- `myohands` — passive anatomical torso scaffold with right and mirrored-left hands derived from pruned arms.
- `myolegs` — passive anatomical torso scaffold with legs.
- `myolegs26` — passive anatomical torso scaffold with reduced-muscle legs and simplified kinematic chain.
- `myolegs_abdomen` — simple abdomen scaffold with legs.
- `myofullbody` — full body with torso, mirrored arms, and legs.

To generate compiled MuJoCo XML files for the primary composed assemblies, run:

```bash
uv run python -m myo_sim.build.compose --generate
```

This writes loadable XML files for `myoarms`, `myotorso`, `myolegs`, `myolegs26`, and `myofullbody` under `myo_sim/models/`. Treat these as generated snapshots for GUI viewing and compatibility; for source edits, update the component XML files and `myo_sim/build/compose.py`, then regenerate. Those downstream XMLs are not actively maintained.

## Install

```bash
pip install myo-sim
```

## Quickstart

```python
import mujoco
import myo_sim

# Load a model by registry name
model, data = myo_sim.load("myolegs")
print(f"Joints: {model.njnt}, Muscles: {model.nu}")

# Or compose another registered model
model, data = myo_sim.load("myotorso")

# Legacy models (backwards compatibility)
model, data = myo_sim.load("myoelbow")
```

```python
# Compose a full-body MjSpec model
from myo_sim.build.compose import build_model
model = build_model("myofullbody")
print(f"Full body — joints: {model.njnt}, muscles: {model.nu}")
```

## Bolt-parity muscles and contact (MuJoCo C + Warp)

Stock MyoSim models use MuJoCo’s rigid-tendon `mjGAIN_MUSCLE` actuators and solver contacts.
Optional De Groote–Fregly muscles and Hunt–Crossley contacts (Bolt-matched) ship as two backends
on the same asset tables (`myo_sim.sprinter_assets`, `myo_sim.fullbody_simple_assets`):

| Backend | Package | How forces are applied |
|---|---|---|
| MuJoCo C (float64) | `myo_sim.plugin` | Passiveive plugin `myo.muscles` inside `mj_step` |
| MuJoCo Warp (float32) | `myo_sim.warp` | `MuscleSystem` / `ContactSystem` → `qfrc_applied` / `xfrc_applied` |

**Build the C plugin** (once per machine / MuJoCo version; needs a C compiler):

```bash
# after pip/uv install, from the package tree — or:
python -c "from myo_sim.plugin import ensure_built; ensure_built()"
# equivalent: bash myo_sim/plugin/build.sh
```

**Warp extras** (optional; not required for core MyoSim):

```bash
pip install warp-lang mujoco-warp
```

**Switch backends with one driver:**

```python
from myo_sim import sprinter_assets
from myo_sim.physics import make_musculoskeletal
import mujoco

muscles = sprinter_assets.load_muscles()
contact = sprinter_assets.load_contact()
limits = sprinter_assets.load_limits()
xml = sprinter_assets.xml_path()
nbody = mujoco.MjModel.from_xml_path(str(xml)).nbody

# CPU / MuJoCo plugin
drv = make_musculoskeletal("c", muscles, contact, limits=limits, nbody=nbody)
m, d = drv.prepare_model(xml)
drv.equilibrate()
drv.step()  # mj_step; plugin applies muscles + contacts

# GPU-batchable Warp path (same assets)
# drv = make_musculoskeletal("warp", muscles, contact, limits=limits, nbody=nbody, device="cuda:0")
# m, d = drv.prepare_model(xml)
# drv.step()  # MuscleSystem + ContactSystem around mjw.step
```

See `NOTICE` for Bolt / msk_envs provenance.

## Development

```bash
git clone https://github.com/MyoHub/myo_sim.git
cd myo_sim
uv sync --dev
uv run pytest tests/ -x -n auto --ignore=tests/test_equivalence.py
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for the full contributor guide.

## Citation

If you find this repository useful in your research, please cite the following works:

```bibtex
@misc{MyoSuite2022,
  author    = {Caggiano, Vittorio and Wang, Huawei and Durandau, Guillaume
               and Sartori, Massimo and Kumar, Vikash},
  title     = {{MyoSuite}: A contact-rich simulation suite for musculoskeletal motor control},
  year      = {2022},
  doi       = {10.48550/ARXIV.2205.13600},
  url       = {https://arxiv.org/abs/2205.13600},
}
```

```bibtex
@article{Li2026MuscleMimic,
  title={Towards Embodied AI with MuscleMimic: Unlocking full-body musculoskeletal motor learning at scale},
  author={Li, Chengkun and Wang, Cheryl and Ziliotto, Bianca and Simos, Merkourios and Kovecses, Jozsef and Durandau, Guillaume and Mathis, Alexander},
  journal={arXiv preprint arXiv:2603.25544},
  year={2026}
}
```

### Musculoskeletal dynamics (De Groote–Fregly muscles, Hunt–Crossley contact)

`myo_sim.warp`, `myo_sim.plugin`, and the `myosprinter` / fullbody_simple bolt-parity assets are derived
from [Bolt](https://github.com/willwng/bolt) ([project page](https://bolt-simulator.github.io); Apache-2.0
since `c1db1fe`) and [msk_envs](https://github.com/willwng/msk_envs) (Apache-2.0 since `398950c`) by
William Wang et al.; see [NOTICE](NOTICE). If you use them, please cite:

```bibtex
@inproceedings{Wang2026Bolt,
  author    = {William Wang and Nicholas Bianco and Guy Tevet and Jennifer Hicks and C. Karen Liu and
               Scott Delp and Kayvon Fatahalian},
  title     = {Learning Realistic Athletic Sprinting Without Demonstrations},
  booktitle = {SIGGRAPH Asia 2026 Conference Papers},
  series    = {SA Conference Papers '26},
  year      = {2026},
  publisher = {Association for Computing Machinery},
  doi       = {10.1145/3829340.3842159},
  url       = {https://doi.org/10.1145/3829340.3842159}
}
```

```bibtex
@article{DeGroote2016,
  author  = {De Groote, Friedl and Kinney, Allison L. and Rao, Anil V. and Fregly, Benjamin J.},
  title   = {Evaluation of Direct Collocation Optimal Control Problem Formulations for Solving the Muscle
             Redundancy Problem},
  journal = {Annals of Biomedical Engineering},
  volume  = {44},
  number  = {10},
  pages   = {2922--2936},
  year    = {2016},
  doi     = {10.1007/s10439-016-1591-9}
}
```

## License

Apache 2.0 — see [LICENSE](LICENSE).
