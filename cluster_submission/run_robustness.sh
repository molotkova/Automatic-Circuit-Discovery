#!/bin/bash

PERTURBATION_TYPE="$1"

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

case "$PERTURBATION_TYPE" in
    "swap_dataset_roles")
        poetry run python -m experiments.robustness.run_experiments \
            --experiment all \
            --perturbation swap_dataset_roles \
            --run-ids oej7igco q1aj4v7a 0gfdvfex 3j7xmd4m m7ocizzd mva8de2j no4ot2lk rg4s225f bpryslek mzuyijib \
            --baseline-ids 4cqitqq2 gkapr9d2 qx86aybd hhb2j8ny h3m3up4z q4i2w4pw yrxtxjdi 1fj8o2b9 1yxiyngt 7u7zmd7q \
            --device cuda \
            --num-examples 100
        ;;
    "add_random_prefixes")
        poetry run python -m experiments.robustness.run_experiments \
            --experiment all \
            --perturbation add_random_prefixes \
            --run-ids 4x4nnkmc 84n391j5 er9ukc3m 1jk51d9n fkla2wkv k2uu5tlh pzlq8fv0 vedrvood xk8u910v e1scwu2z \
            --baseline-ids gkapr9d2 qx86aybd hhb2j8ny h3m3up4z 4cqitqq2 q4i2w4pw yrxtxjdi 1fj8o2b9 1yxiyngt 7u7zmd7q \
            --device cuda \
            --num-examples 100
        ;;
    "shuffle_abc_prompts")
        poetry run python -m experiments.robustness.run_experiments \
            --experiment all \
            --perturbation shuffle_abc_prompts \
            --run-ids 2cmlgn4z 9woq5yej pozq1kcl uuywclwi 1cjeqiq7 285k2zbj gp4litlj m278sb60 m71uvccm xm64deab \
            --baseline-ids gkapr9d2 qx86aybd hhb2j8ny h3m3up4z 4cqitqq2 q4i2w4pw yrxtxjdi 1fj8o2b9 1yxiyngt 7u7zmd7q \
            --device cuda \
            --num-examples 100
        ;;
    "None")
        poetry run python -m experiments.robustness.run_experiments \
            --experiment all \
            --perturbation None \
            --run-ids gkapr9d2 qx86aybd hhb2j8ny h3m3up4z 4cqitqq2 q4i2w4pw yrxtxjdi 1fj8o2b9 1yxiyngt 7u7zmd7q \
            --baseline-ids gkapr9d2 qx86aybd hhb2j8ny h3m3up4z 4cqitqq2 q4i2w4pw yrxtxjdi 1fj8o2b9 1yxiyngt 7u7zmd7q \
            --device cuda \
            --num-examples 100
        ;;
esac