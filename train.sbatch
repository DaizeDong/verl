#!/bin/bash
#SBATCH --job-name=verl-train
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=raisl
#SBATCH --account=raisl
#SBATCH --gpus=8
#SBATCH --cpus-per-task=192
#SBATCH --mem=1000000
#SBATCH --time=24:00:00
#SBATCH --output=logs/verl-train-%j.out
#SBATCH --error=logs/verl-train-%j.err

set -euo pipefail

mkdir -p logs

# Adjust these paths if needed.
WORKDIR="/common/users/jc3585/verl"
SIF_PATH="/common/users/jc3585/env/verlsgl055.sif"

singularity exec --nv "${SIF_PATH}" bash -lc "cd \"${WORKDIR}\" && bash train.sh"
