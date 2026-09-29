#!/bin/bash -l
# One A100 job that runs a Python module of the repository, with every stage in the Apptainer images
#   sbatch picasso/job.sbatch evaluation.run --dataset replica --scene room_0 ...
# As an array, each task adds --only-scene with the --scene value at its index, so the scenes of
# an experiment run in parallel:
#   sbatch --array=0-6 picasso/job.sbatch evaluation.scripts.experiment --experiment validation --scene ...
# Submit it from $FSCRATCH/tfg, where the logs directory is
#SBATCH --job-name=tfg
#SBATCH --partition=gpu_partition
#SBATCH --gres=gpu:1
#SBATCH --constraint=a100
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
#SBATCH --output=logs/%x-%j.out

source "${TFG_REPO:-$FSCRATCH/tfg/repo}/picasso/env.sh"
set -eo pipefail

arguments=("$@")
if [ -n "$SLURM_ARRAY_TASK_ID" ]; then
    # Collect the --scene values and keep the one at the index of this task
    scenes=()
    for ((i = 1; i < ${#arguments[@]}; i++)); do
        if [ "${arguments[i - 1]}" = "--scene" ]; then
            scenes+=("${arguments[i]}")
        fi
    done
    arguments+=(--only-scene "${scenes[SLURM_ARRAY_TASK_ID]}")
fi

echo "$(hostname): $(nvidia-smi -L)"
cd "$TFG_REPO"
"$TFG_PYTHON" -m "${arguments[@]}"

# Scannet++ reads the resized DSLR images only to undistort them, so once a scene has its
# undistorted COLMAP model they are removed to save files of the fscratch quota
for dslr in "$TFG_DATA"/scannetpp/validation_data/*/dslr; do
    sparse="$dslr/undistorted_colmap/sparse/0"
    if [ -e "$sparse/points3D.bin" ] || [ -e "$sparse/points3D.txt" ]; then
        rm -rf "$dslr/resized_images"
    fi
done
