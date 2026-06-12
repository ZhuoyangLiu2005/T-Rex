#!/bin/bash
# Source this in every terminal before running the teleop stack:
#   source setup.sh
# Override CONDA_ENV_NAME / ROBOT_NAME / ROBOT_IP in your environment as needed.

# Activate the Python environment: prefer a uv/venv .venv at the repo root,
# fall back to the conda environment.
_SETUP_SH_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -f "${_SETUP_SH_DIR}/.venv/bin/activate" ]; then
    source "${_SETUP_SH_DIR}/.venv/bin/activate"
else
    conda activate "${CONDA_ENV_NAME:-sharpa-dexmate-tmp}"
fi
unset _SETUP_SH_DIR

# dexcontrol robot identity — set these to match your robot.
export ROBOT_NAME="${ROBOT_NAME:-dm/vgd1262ab823-1p}"
export ROBOT_IP="${ROBOT_IP:-192.168.50.20}"

# Make the Sharpa Wave SDK python module importable if installed at the
# default location (see third_party/README.md).
if [ -d /opt/sharpa-wave-sdk/python ]; then
    export PYTHONPATH="/opt/sharpa-wave-sdk/python:${PYTHONPATH}"
fi

# Kill any process bound to the tactile ports (50001, 50002).
for port in 50001 50002; do
    pids=$(sudo lsof -ti :"$port" 2>/dev/null | sort -u)
    if [ -n "$pids" ]; then
        echo "Killing PID(s) on port $port: $pids"
        sudo kill -9 $pids
    else
        echo "No process on port $port"
    fi
done
unset port pids
