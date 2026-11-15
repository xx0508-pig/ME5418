#!/usr/bin/env bash
# Run the project's private Python without changing the current shell.
set -euo pipefail
PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"
if [[ ! -x .venv/bin/python ]]; then
  echo 'Missing .venv. Create it with: conda env create -p "$PWD/.venv" -f environment.yml' >&2
  exit 1
fi
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$PROJECT_DIR/.local/cache}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-$PROJECT_DIR/.local/matplotlib}"
export HF_HOME="${HF_HOME:-$PROJECT_DIR/.local/huggingface}"
export LIBERO_CONFIG_PATH="${LIBERO_CONFIG_PATH:-$PROJECT_DIR/.local/libero-config}"
export NUMBA_CACHE_DIR="${NUMBA_CACHE_DIR:-$PROJECT_DIR/.local/numba}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYTEST_DISABLE_PLUGIN_AUTOLOAD="${PYTEST_DISABLE_PLUGIN_AUTOLOAD:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
# A Mesa fallback allows local debugging when the NVIDIA driver is unavailable.
# Explicit caller settings always win.
if [[ "$MUJOCO_GL" == egl && -f /usr/share/glvnd/egl_vendor.d/50_mesa.json ]]; then
  if ! command -v nvidia-smi >/dev/null 2>&1 || ! timeout 5 nvidia-smi >/dev/null 2>&1; then
    export __EGL_VENDOR_LIBRARY_FILENAMES="${__EGL_VENDOR_LIBRARY_FILENAMES:-/usr/share/glvnd/egl_vendor.d/50_mesa.json}"
    export LIBGL_ALWAYS_SOFTWARE="${LIBGL_ALWAYS_SOFTWARE:-1}"
  fi
fi
exec "$PROJECT_DIR/.venv/bin/python" "$@"
