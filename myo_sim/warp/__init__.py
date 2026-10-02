"""mujoco_warp-native musculoskeletal physics (DGF muscle contraction, Hunt-Crossley contact).

Optional submodule -- importing myo_sim.warp requires `warp-lang` and `mujoco-warp` (not core myo_sim
dependencies). Keep `import myo_sim` free of a Warp/mujoco_warp requirement; only touch this package when a
consumer actually needs GPU-batched simulation (e.g. an mjlab task).
"""

from __future__ import annotations

from myo_sim.warp.contact_system import ContactSystem as ContactSystem
from myo_sim.warp.muscle_system import MuscleSystem as MuscleSystem
