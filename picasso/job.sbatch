#!/bin/bash -l
# Job on one A100 that runs a Python module of the repository, with every stage in the Apptainer images
#   sbatch picasso/job.sbatch evaluation.run --dataset replica --scene room_0 ...
# picasso/submit.sh uses it to submit the steps of the evaluation, with one job per scene.
# It is necessary to submit it from $FSCRATCH/tfg, which is where the logs directory is
#
# The jobs of up to two hours in the short QoS start much sooner, because SLURM fits them in the gaps
# left by the larger ones. The work stops TFG_MARGIN seconds before the end of the job with exit code 3,
# and the job submits itself again with the same command, which continues from the cached stages and
# from the resume checkpoint of the training
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

# The margin leaves time to save the checkpoint of the training before SLURM stops the job. The fallback
# reads scontrol, because the squeue of Picasso is a wrapper that does not handle -o reliably
end=${SLURM_JOB_END_TIME:-$(date -d "$(scontrol show job "$SLURM_JOB_ID" | grep -o 'EndTime=[^ ]*' | cut -d= -f2)" +%s)}
export TFG_DEADLINE=$((end - ${TFG_MARGIN:-600}))
part=${TFG_PART:-1}
echo "$(hostname): $(nvidia-smi -L), part $part, stops at $(date -d "@$TFG_DEADLINE")"

cd "$TFG_REPO"
code=0
"$TFG_PYTHON" -m "${arguments[@]}" || code=$?

# Scannet++ reads the resized DSLR images only in order to undistort them. For this reason, once a scene
# named in the arguments of this job has its undistorted COLMAP model, they are removed to save files of
# the fscratch quota. Note that the scenes of other jobs are never touched
for argument in "${arguments[@]}"; do
    dslr="$TFG_DATA/scannetpp/validation_data/$argument/dslr"
    [ -n "$argument" ] && [ -d "$dslr" ] || continue
    sparse="$dslr/undistorted_colmap/sparse/0"
    if [ -e "$sparse/points3D.bin" ] || [ -e "$sparse/points3D.txt" ]; then
        rm -rf "$dslr/resized_images"
    fi
done

# Only exit code 3 continues in a new job, never an error. In addition, a chain stops after TFG_MAX_PARTS
# jobs, so that a run that makes no progress does not submit itself forever
if [ "$code" -eq 3 ]; then
    if [ "$part" -ge "${TFG_MAX_PARTS:-12}" ]; then
        echo "stopped: part $part reached TFG_MAX_PARTS, submit the same command again to continue"
        exit 3
    fi

    # The new job takes its time and QoS from the directives of this file, and its memory from
    # SBATCH_MEM_PER_NODE in the case that the first job was submitted with it. This is due to the fact
    # that the job inherits the environment. It starts once this one has ended
    TFG_PART=$((part + 1)) sbatch \
        --job-name="$SLURM_JOB_NAME" \
        --chdir="$SLURM_SUBMIT_DIR" --dependency=afterany:"$SLURM_JOB_ID" \
        "$TFG_REPO/picasso/job.sbatch" "${arguments[@]}"
    echo "continues in part $((part + 1))"
    exit 0
fi
exit $code
