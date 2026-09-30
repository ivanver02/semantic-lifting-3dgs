#!/bin/bash -l
# One A100 job that runs a Python module of the repository, with every stage in the Apptainer images
#   sbatch picasso/job.sbatch evaluation.run --dataset replica --scene room_0 ...
# picasso/submit.sh submits the steps of the evaluation with it, one job per scene.
# Submit it from $FSCRATCH/tfg, where the logs directory is
#
# Jobs of up to two hours in the short QoS start much sooner, as SLURM fits them in the gaps it
# leaves free. Work stops TFG_MARGIN seconds before the end of the job with exit code 3, and the
# job submits itself again with the same command, which continues from the cached stages and the
# training resume checkpoint
#SBATCH --job-name=tfg
#SBATCH --partition=gpu_partition
#SBATCH --qos=short
#SBATCH --gres=gpu:1
#SBATCH --constraint=a100
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x-%j.out

source "${TFG_REPO:-$FSCRATCH/tfg/repo}/picasso/env.sh"
set -eo pipefail

arguments=("$@")

# The margin leaves time to save the training checkpoint before SLURM stops the job. The fallback
# reads scontrol, since the squeue of Picasso is a wrapper that does not handle -o reliably
end=${SLURM_JOB_END_TIME:-$(date -d "$(scontrol show job "$SLURM_JOB_ID" | grep -o 'EndTime=[^ ]*' | cut -d= -f2)" +%s)}
export TFG_DEADLINE=$((end - ${TFG_MARGIN:-600}))
part=${TFG_PART:-1}
echo "$(hostname): $(nvidia-smi -L), part $part, stops at $(date -d "@$TFG_DEADLINE")"

cd "$TFG_REPO"
code=0
"$TFG_PYTHON" -m "${arguments[@]}" || code=$?

# Scannet++ reads the resized DSLR images only to undistort them, so once a scene named in the
# arguments of this job has its undistorted COLMAP model they are removed to save files of the
# fscratch quota. Scenes of other jobs are never touched
for argument in "${arguments[@]}"; do
    dslr="$TFG_DATA/scannetpp/validation_data/$argument/dslr"
    [ -n "$argument" ] && [ -d "$dslr" ] || continue
    sparse="$dslr/undistorted_colmap/sparse/0"
    if [ -e "$sparse/points3D.bin" ] || [ -e "$sparse/points3D.txt" ]; then
        rm -rf "$dslr/resized_images"
    fi
done

# Only exit code 3 continues in a new job, never an error, and a chain stops after TFG_MAX_PARTS
# jobs so a run that makes no progress does not submit itself forever
if [ "$code" -eq 3 ]; then
    if [ "$part" -ge "${TFG_MAX_PARTS:-12}" ]; then
        echo "stopped: part $part reached TFG_MAX_PARTS, submit the same command again to continue"
        exit 3
    fi

    # The new job takes its time and QoS from the directives of this file, and its memory from
    # SBATCH_MEM_PER_NODE when the first job was submitted with it, as the job inherits the
    # environment. It starts once this one has ended
    TFG_PART=$((part + 1)) sbatch \
        --job-name="$SLURM_JOB_NAME" \
        --chdir="$SLURM_SUBMIT_DIR" --dependency=afterany:"$SLURM_JOB_ID" \
        "$TFG_REPO/picasso/job.sbatch" "${arguments[@]}"
    echo "continues in part $((part + 1))"
    exit 0
fi
exit $code
