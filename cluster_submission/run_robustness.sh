#!/bin/bash
# Unset PYTHONPATH to avoid any local Python settings
unset PYTHONPATH
# Unset any conda-related environment variables to start fresh
unset CONDA_PREFIX
unset CONDA_DEFAULT_ENV
unset CONDA_PYTHON_EXE
unset CONDA_SHLVL

# Set torch visible gpus environment variable
if command -v nvidia-smi &> /dev/null; then
    GPU_COUNT=$(nvidia-smi --list-gpus | wc -l)
    if [ "$GPU_COUNT" -gt 0 ]; then
        echo "Detected $GPU_COUNT GPUs"
        # Create a comma-separated list of all GPU indices
        GPU_INDICES=$(seq -s, 0 $((GPU_COUNT-1)))
        export CUDA_VISIBLE_DEVICES=$GPU_INDICES
        echo "Setting CUDA_VISIBLE_DEVICES=$GPU_INDICES"
    else
        echo "No GPUs detected"
    fi
else
    echo "nvidia-smi not found, assuming no GPUs available"
fi

path=/home/somo00003/Automatic-Circuit-Discovery  # Project working directory

source ~/miniconda3/etc/profile.d/conda.sh
conda activate acdc-env

cd "${path}"
pwd=$(pwd)

export HOME="$(pwd)"
export PYTHONPATH="${path}"
export PYTHONUNBUFFERED=1

poetry config virtualenvs.in-project true

# Verify the environment is activated
echo "Current Python: $(which python)"
echo "Environment info:"
poetry env info

poetry run python -m experiments.robustness.run_experiments \
    --experiment all \
    --run-ids ud8s5c37 cl3nhhvp \
    --baseline-id sjr6k1ip \
    --device cuda \
    --num-examples 100
