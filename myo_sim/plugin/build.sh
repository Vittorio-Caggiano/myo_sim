#!/bin/bash
# Build the C libraries against the MuJoCo shipped inside the installed `mujoco` Python package (no
# vendored MuJoCo SDK needed).
#   libmyo_muscle.so   muscle spec (ctypes)                    libmyo_contact.so  Hunt-Crossley contacts (ctypes)
#   libmyo_plugin.so   MuJoCo passive plugin "myo.muscles" (muscles, limits, contacts)
# Linux and macOS. On Windows use WSL2.
set -euo pipefail
cd "$(dirname "$0")"
PY=${PYTHON:-python}
MJ=$($PY -c "import mujoco, os; print(os.path.dirname(mujoco.__file__))")
MJVER=$($PY -c "import mujoco; print(mujoco.__version__)")
CC=${CC:-cc}
CFLAGS="-O2 -fPIC -ffp-contract=off -Wall -Wno-unused-function"
case "$(uname -s)" in
  Linux)
    LIB=$(ls "$MJ"/libmujoco.so.$MJVER 2>/dev/null || ls "$MJ"/libmujoco.so* | head -1)
    SHARED="-shared"; RPATH="-Wl,-rpath,$MJ" ;;
  Darwin)
    LIB=$(ls "$MJ"/libmujoco*.dylib 2>/dev/null | head -1 || true)
    if [ -z "$LIB" ]; then   # some wheels ship a framework instead of a flat dylib
      LIB=$(find "$MJ" -name "mujoco" -path "*MuJoCo.framework*" -type f | head -1)
    fi
    SHARED="-dynamiclib -undefined dynamic_lookup"; RPATH="-Wl,-rpath,$MJ" ;;
  *) echo "unsupported OS $(uname -s); use Linux, macOS or WSL2" >&2; exit 1 ;;
esac
INC="$MJ/include"
[ -d "$INC" ] || { echo "MuJoCo headers not found in $INC" >&2; exit 1; }
echo "MuJoCo $MJVER: $LIB"
$CC $CFLAGS $SHARED -o libmyo_muscle.so  myo_muscle.c  -lm
$CC $CFLAGS $SHARED -o libmyo_contact.so myo_contact.c -lm
$CC $CFLAGS $SHARED -I"$INC" -o libmyo_plugin.so myo_plugin.c myo_muscle.c myo_contact.c "$LIB" $RPATH -lm
echo "built myo_sim/plugin/libmyo_muscle.so libmyo_contact.so libmyo_plugin.so"
