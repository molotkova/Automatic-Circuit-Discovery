#!/bin/bash
# Unset PYTHONPATH to avoid any local Python settings
unset PYTHONPATH
# Unset any conda-related environment variables to start fresh
unset CONDA_PREFIX
unset CONDA_DEFAULT_ENV
unset CONDA_PYTHON_EXE
unset CONDA_SHLVL

path=/home/somo00003/Automatic-Circuit-Discovery  # Project working directory

source ~/miniconda3/etc/profile.d/conda.sh
conda activate acdc-env

cd "${path}"
pwd=$(pwd)

export HOME="$(pwd)"
export PYTHONPATH="${path}"
export PYTHONUNBUFFERED=1

# Install dependicies once if not already installed
# poetry config virtualenvs.in-project true
# poetry env use $(which python)
# poetry install
# poetry show

poetry run python cluster_submission/test_hello_world_sleep.py