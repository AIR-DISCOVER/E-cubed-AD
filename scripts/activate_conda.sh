#!/usr/bin/env bash
# Source this file after setting REPO_ROOT to activate the project environment.

_dd_conda_env="${CONDA_ENV_NAME:-diffusiondrive_nusc}"
if [[ -n "${CONDA_PREFIX:-}" && "${CONDA_DEFAULT_ENV:-}" == "$_dd_conda_env" ]]; then
    unset _dd_conda_env
    return 0
fi

if [[ -n "${CONDA_EXE:-}" && -x "${CONDA_EXE}" ]]; then
    _dd_conda="${CONDA_EXE}"
elif command -v conda >/dev/null 2>&1; then
    _dd_conda="$(command -v conda)"
else
    echo "Conda is not available on PATH. Add Conda to PATH or set CONDA_EXE." >&2
    unset _dd_conda_env
    return 1
fi

eval "$("$_dd_conda" shell.bash hook)"
conda activate "$_dd_conda_env"
unset _dd_conda _dd_conda_env
