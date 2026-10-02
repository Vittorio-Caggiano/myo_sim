"""ctypes bindings to the C implementation + helpers to author the myo.muscles plugin spec and read its state.

Shared libraries are built by ``ensure_built()`` / ``build.sh``; ctypes loads are deferred until first use
so ``import myo_sim.plugin`` does not require a prior build. Muscle/contact formulation follows Bolt
(Apache-2.0); see NOTICE.
"""

from __future__ import annotations

import ctypes
import os
import pathlib
import subprocess
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
PLUGIN_LIB = str(HERE / "libmyo_plugin.so")
NOUT = 12
NSTATE = 10
OUT_KEYS = [
    "tendon_force",
    "norm_fiber_length",
    "norm_fiber_velocity",
    "fiber_velocity",
    "active_fiber_force",
    "fal",
    "fv",
    "fpe",
    "pennation",
    "ln_dot",
    "tendon_length",
    "clamped",
]
STATE_KEYS = [
    "act",
    "ln",
    "force",
    "length",
    "velocity",
    "act_dot",
    "ln_dot",
    "metabolic",
    "fiber_velocity",
    "norm_fiber_length",
]
MET_BHARGAVA, MET_NEG_WORK, MET_FORBID_NEG, MET_MIN_HEAT, MET_BOLT = 1, 2, 4, 8, 16
MET_OFF = 256  # plugin only: skip metabolics (slot stays 0); dynamics unchanged
MET_DEFAULT = MET_BHARGAVA | MET_NEG_WORK | MET_FORBID_NEG | MET_MIN_HEAT

_D = ctypes.POINTER(ctypes.c_double)
_lib = None
_loaded = False


def _plugin_lib_path() -> pathlib.Path:
    """Platform-aware path to the MuJoCo plugin shared library."""
    if sys.platform == "darwin":
        # build.sh still emits .so on macOS via -dynamiclib -o libmyo_plugin.so
        p = HERE / "libmyo_plugin.so"
        if p.exists():
            return p
        dylib = HERE / "libmyo_plugin.dylib"
        if dylib.exists():
            return dylib
        return p
    return HERE / "libmyo_plugin.so"


def _linked_mujoco_hint(plugin: pathlib.Path) -> str | None:
    """Return the mujoco dylib id the plugin was linked against, if discoverable."""
    try:
        if sys.platform == "darwin":
            out = subprocess.run(["otool", "-L", str(plugin)], capture_output=True, text=True, check=False)
            for line in out.stdout.splitlines()[1:]:
                line = line.strip().split(" ", 1)[0]
                if "libmujoco" in line or "mujoco.framework" in line:
                    return line
        else:
            out = subprocess.run(["ldd", str(plugin)], capture_output=True, text=True, check=False)
            for line in out.stdout.splitlines():
                if "libmujoco" in line:
                    return line.strip()
    except OSError:
        return None
    return None


def _plugin_matches_current_mujoco(plugin: pathlib.Path) -> bool:
    """True if the existing plugin .so looks linked to this process's mujoco package."""
    try:
        import mujoco
    except ImportError:
        return False
    hint = _linked_mujoco_hint(plugin)
    if not hint:
        return True  # cannot tell; assume ok
    ver = mujoco.__version__
    mj_dir = pathlib.Path(mujoco.__file__).resolve().parent
    # Version token in the linked name must match, and rpath dir should be under this install when absolute.
    if ver not in hint and f"libmujoco.{ver}" not in hint:
        # allow "libmujoco.so.3.9.0" style / framework path containing version
        if ver.split(".")[0] not in hint:
            return False
        # still require major.minor presence when possible
        mm = ".".join(ver.split(".")[:2])
        if mm not in hint and ver not in hint:
            return False
    if hint.startswith("/") and str(mj_dir) not in hint:
        return False
    return True


def ensure_built(*, force: bool = False) -> pathlib.Path:
    """Build ``libmyo_*.so`` via ``build.sh`` if missing, stale, or ``force``.

    Returns the path to ``libmyo_plugin.so``. Raises RuntimeError with a clear message if the
    C toolchain or MuJoCo headers are unavailable.
    """
    global PLUGIN_LIB
    plugin = _plugin_lib_path()
    muscle = HERE / "libmyo_muscle.so"
    if not force and plugin.exists() and muscle.exists() and _plugin_matches_current_mujoco(plugin):
        PLUGIN_LIB = str(plugin)
        return plugin
    build = HERE / "build.sh"
    if not build.is_file():
        raise RuntimeError(f"plugin build script missing: {build}")
    env = os.environ.copy()
    env.setdefault("PYTHON", sys.executable)
    try:
        subprocess.run(
            ["bash", str(build)],
            cwd=str(HERE),
            check=True,
            env=env,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError as e:
        raise RuntimeError(
            "Cannot build myo_sim.plugin: bash/C compiler not found. "
            "Install a C toolchain and re-run myo_sim.plugin.ensure_built()."
        ) from e
    except subprocess.CalledProcessError as e:
        msg = (e.stderr or e.stdout or "").strip() or str(e)
        raise RuntimeError(
            "Failed to build myo_sim.plugin shared libraries. "
            "Need a C compiler and the `mujoco` Python package (headers under the wheel). "
            f"build.sh said:\n{msg}"
        ) from e
    plugin = _plugin_lib_path()
    if not plugin.exists():
        raise RuntimeError(f"build.sh finished but {plugin.name} is still missing under {HERE}")
    PLUGIN_LIB = str(plugin)
    # Clear cached ctypes handle so the next _get_lib picks up the new muscle .so
    global _lib
    _lib = None
    return plugin


def _get_lib():
    global _lib
    if _lib is None:
        ensure_built()
        muscle = HERE / "libmyo_muscle.so"
        if not muscle.exists():
            raise RuntimeError(f"libmyo_muscle.so missing after ensure_built() under {HERE}")
        _lib = ctypes.CDLL(str(muscle))
    return _lib


def _p(a):
    a = np.ascontiguousarray(a, dtype=np.float64)
    return a, a.ctypes.data_as(_D)


def curves(x, ws, lmin, e0, strain):
    n = len(x)
    out = np.zeros((n, 7))
    args = [_p(v) for v in (x, ws, lmin, e0, strain)]
    _get_lib().myo_curves_batch(ctypes.c_int(n), *[a[1] for a in args], out.ctypes.data_as(_D))
    return dict(zip(["fl", "fv", "fv_inv", "fp", "ft", "ft_inv", "fv_deriv"], out.T))


def act_dot(a, u, p):
    n = len(a)
    out = np.zeros(n)
    A, U, Pp = _p(a), _p(u), _p(p)
    _get_lib().myo_act_dot_batch(ctypes.c_int(n), A[1], U[1], Pp[1], out.ctypes.data_as(_D))
    return out


def contraction(L, V, a, ln, p):
    n = len(L)
    out = np.zeros((n, NOUT))
    args = [_p(v) for v in (L, V, a, ln, p)]
    _get_lib().myo_contraction_batch(ctypes.c_int(n), *[x[1] for x in args], out.ctypes.data_as(_D))
    d = dict(zip(OUT_KEYS, out.T))
    d["_raw"] = out
    return d


def equilibrium(L, V, a, p):
    n = len(L)
    out = np.zeros(n)
    args = [_p(v) for v in (L, V, a, p)]
    _get_lib().myo_equilibrium_batch(ctypes.c_int(n), *[x[1] for x in args], out.ctypes.data_as(_D))
    return out


def metabolics(a, u, cout, p, flags=MET_DEFAULT, aerobic=1.0, effort=1.0):
    n = len(a)
    out = np.zeros(n)
    args = [_p(v) for v in (a, u, cout["_raw"], p)]
    _get_lib().myo_metabolics_batch(
        ctypes.c_int(n),
        *[x[1] for x in args],
        ctypes.c_int(flags),
        ctypes.c_double(aerobic),
        ctypes.c_double(effort),
        out.ctypes.data_as(_D),
    )
    return out


def poly(Q, coef, dim, order):
    n = Q.shape[0]
    L = np.zeros(n)
    G = np.zeros((n, dim))
    Qc, Cc = _p(Q), _p(coef)
    _get_lib().myo_poly_batch(
        ctypes.c_int(n), ctypes.c_int(dim), ctypes.c_int(order), Cc[1], Qc[1], L.ctypes.data_as(_D), G.ctypes.data_as(_D)
    )
    return L, G


# ------------------------------------------------------------------------------------------- plugin authoring
def write_spec(path, muscles, met_flags=MET_DEFAULT, aerobic=1.0, effort=1.0, limits=(), contact=None):
    """muscles: list of dicts with keys p (NPARAM), family (0 DGF, 1 MuJoCo), ctrl_id, path ('poly'|'tendon') and
    either (dim, order, qposadr, dofadr, coef) or tendon_id; MuJoCo family also gainprm, lengthrange, acc0, dynprm."""
    lines = [f"{len(muscles)} {met_flags} {aerobic!r} {effort!r}"]
    for m in muscles:
        tok = [repr(float(v)) for v in m["p"]]
        ptype = {"poly": 0, "tendon": 1, "poly+tendon": 2}[m["path"]]
        tok += [str(m["family"]), str(m["ctrl_id"]), str(ptype)]
        if ptype in (0, 2):
            tok += (
                [str(m["dim"]), str(m["order"])]
                + [str(int(v)) for v in m["qposadr"]]
                + [str(int(v)) for v in m["dofadr"]]
                + [repr(float(v)) for v in m["coef"]]
            )
        if ptype in (1, 2):
            tok += [str(m["tendon_id"])]
        if m["family"] == 1:
            tok += (
                [repr(float(v)) for v in m["gainprm"][:9]]
                + [repr(float(v)) for v in m["lengthrange"]]
                + [repr(float(m["acc0"]))]
                + [repr(float(v)) for v in m["dynprm"][:3]]
            )
        lines.append(" ".join(tok))
    lines.append(str(len(limits)))
    for lim in limits:  # dict(qposadr, dofadr, q_low, q_up, k_low, k_up, damping, transition)
        lines.append(
            " ".join(
                [str(int(lim["qposadr"])), str(int(lim["dofadr"]))]
                + [repr(float(lim[k])) for k in ("q_low", "q_up", "k_low", "k_up", "damping", "transition")]
            )
        )
    if contact is not None:  # dict(geom_flat=(ngeom, 26) with simulator body ids, pairs=(npair, 2), naconmax)
        gf = np.asarray(contact["geom_flat"], np.float64)
        pr = np.asarray(contact["pairs"], int).reshape(-1, 2)
        lines.append(f"{gf.shape[0]} {pr.shape[0]} {int(contact['naconmax'])}")
        lines += [" ".join(repr(float(v)) for v in row) for row in gf]
        lines += [f"{a} {b}" for a, b in pr]
    pathlib.Path(path).write_text("\n".join(lines) + "\n")


def plugin_xml(spec_path, instance="muscles"):
    return (
        f'<extension><plugin plugin="myo.muscles"><instance name="{instance}">'
        f'<config key="spec" value="{spec_path}"/></instance></plugin></extension>'
    )


def inject_plugin_xml(xml_text: str, spec_path: str, instance: str = "muscles") -> str:
    """Insert the myo.muscles extension before the first ``<worldbody>`` if not already present."""
    if 'plugin="myo.muscles"' in xml_text or "plugin='myo.muscles'" in xml_text:
        return xml_text
    marker = "<worldbody>"
    if marker not in xml_text:
        raise ValueError("MJCF text has no <worldbody>; cannot inject myo.muscles plugin")
    return xml_text.replace(marker, plugin_xml(spec_path, instance=instance) + "\n  " + marker, 1)


def contact_state_view(mjm, mjd, n, instance=0):
    adr = mjm.plugin_stateadr[instance] + NSTATE * n
    return mjd.plugin_state[adr : adr + 5]  # grf xyz, nacon, ncollision


def state_view(mjm, mjd, n, instance=0):
    adr = mjm.plugin_stateadr[instance]
    return mjd.plugin_state[adr : adr + NSTATE * n].reshape(n, NSTATE)


def load_plugin():
    global _loaded, PLUGIN_LIB
    if _loaded:
        return
    import mujoco

    ensure_built()
    PLUGIN_LIB = str(_plugin_lib_path())
    try:
        mujoco.mj_loadPluginLibrary(PLUGIN_LIB)
    except Exception:
        # Stale binary linked against another MuJoCo install/version — rebuild once.
        ensure_built(force=True)
        PLUGIN_LIB = str(_plugin_lib_path())
        mujoco.mj_loadPluginLibrary(PLUGIN_LIB)
    _loaded = True
