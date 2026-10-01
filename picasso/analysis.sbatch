#!/bin/bash -l
# The two analyses that read the masks and the cached votes, once the test has finished. The first one
# compares YOLO with the annotation masks in 2D next to the 3D result, and the second one computes three
# scores for selecting the Gaussians. Both run in lifting.sif with the data read-only, and they write
# their CSVs to $TFG_DATA/analysis
#   bash picasso/submit.sh analysis
#SBATCH --job-name=analysis
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
mkdir -p "$TFG_DATA/analysis"

run() {
    apptainer exec --cleanenv --home /tmp \
        --bind "$TFG_REPO:/repo:ro" --bind "$TFG_DATA:/data:ro" --bind "$TFG_DATA/analysis:/analysis:rw" \
        --pwd /repo "$TFG_SIF_DIR/lifting.sif" python -m "$@"
}

echo "================ 2D masks against 3D results"
run evaluation.scripts.mask_agreement --data /data --output /analysis/mask_agreement.csv
echo
echo "================ scores for selecting Gaussians"
run evaluation.scripts.threshold_scores --data /data --output /analysis/threshold_scores.csv
