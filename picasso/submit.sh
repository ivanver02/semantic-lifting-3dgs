#!/bin/bash
# Submit each step of the evaluation, with one job per scene wherever the scenes are independent
#   bash picasso/submit.sh development    tau phase of the two development scenes, one job each,
#                                         or of the scenes given after it
#   bash picasso/submit.sh warmup         model, masks and votes of the validation scenes, one job each,
#                                         which validation and contribution reuse, outside the analytics
#   bash picasso/submit.sh sweep          rest of the development sweep and the tau and theta selection
#   bash picasso/submit.sh validation     the seven validation scenes, one job each
#   bash picasso/submit.sh selection      validation selection, run here as it only reads the analytics
#   bash picasso/submit.sh test           the ten Scannet++ test scenes, one job each
#   bash picasso/submit.sh contribution   contribution analysis of the validation scenes, one job each
# Each step starts once every job of the previous one has finished, which squeue -u $USER shows

source "$(dirname "$0")/env.sh"
set -eo pipefail

VALIDATION_SCENES=(office_1 office_2 office_3 office_4 room_0 room_1 room_2)
TEST_SCENES=()  # The ten Scannet++ test scenes, still to be chosen
DEVELOPMENT_SCENES=(office_0 7831862f02)
BETAS=(0.50 0.70 0.90 0.94 0.95 0.96 0.97 0.975 0.98 0.985 0.99 0.995 0.999)
TAU_GRID=(0.02 0.03 0.05 0.08 0.10)
THETA_GRID=(0.3 0.4 0.5 0.6 0.7)

ANALYTICS=$TFG_DATA/analytics
DEVELOPMENT_SELECTION=$ANALYTICS/tau_theta_selection.json
SELECTION=$ANALYTICS/selection.json

DEVELOPMENT_SWEEP=(
    evaluation.scripts.development_sweep
    --replica-data-root "$TFG_DATA/replica" --scannetpp-data-root "$TFG_DATA/scannetpp"
    --replica-scene "${DEVELOPMENT_SCENES[0]}" --scannetpp-scene "${DEVELOPMENT_SCENES[1]}"
    --tau-grid "${TAU_GRID[@]}" --theta-grid "${THETA_GRID[@]}"
)

# Jobs write their logs relative to the directory they are submitted from
mkdir -p "$TFG_ROOT/logs"
cd "$TFG_ROOT"

submit() {
    # Submit one job named after its step and scene, so its log is logs/<name>-<job id>.out
    local name=$1
    shift
    sbatch --job-name="$name" "$TFG_REPO/picasso/job.sbatch" "$@"
}

experiment() {
    # Submit one job per scene of an experiment, each with the full scene list and its own --only-scene
    local name=$1 dataset=$2 selection=$3
    shift 3
    if [ ! -e "$selection" ]; then
        echo "missing $selection, the previous step has not finished"
        exit 1
    fi
    local scenes=() scene
    for scene in "$@"; do
        scenes+=(--scene "$scene")
    done
    for scene in "$@"; do
        submit "$name-$scene" evaluation.scripts.experiment --experiment "$name" --selection "$selection" \
            --data-root "$TFG_DATA/$dataset" --output-root "$TFG_DATA/$dataset/eval" \
            "${scenes[@]}" --only-scene "$scene"
    done
}

case "$1" in
    development)
        scenes=("${@:2}")
        if [ ${#scenes[@]} -eq 0 ]; then
            scenes=("${DEVELOPMENT_SCENES[@]}")
        fi
        for scene in "${scenes[@]}"; do
            submit "development-$scene" "${DEVELOPMENT_SWEEP[@]}" --only-scene "$scene"
        done
        ;;
    warmup)
        # The votes do not depend on tau, theta or gamma, so this run with the default configuration
        # fills the same output directories that the validation units read, without recording anything
        for scene in "${VALIDATION_SCENES[@]}"; do
            submit "warmup-$scene" evaluation.run --dataset replica --scene "$scene" \
                --data-root "$TFG_DATA/replica" --output-root "$TFG_DATA/replica/eval/replica/$scene" \
                --variant warmup --mask-source both --betas "${BETAS[@]}"
        done
        ;;
    sweep)
        submit development-sweep "${DEVELOPMENT_SWEEP[@]}"
        ;;
    validation)
        experiment validation replica "$DEVELOPMENT_SELECTION" "${VALIDATION_SCENES[@]}"
        ;;
    selection)
        scenes=()
        for scene in "${VALIDATION_SCENES[@]}"; do
            scenes+=(--scene "$scene")
        done
        cd "$TFG_REPO"
        "$TFG_PYTHON" -m evaluation.scripts.selection --analytics "$ANALYTICS" --output "$SELECTION" \
            --development-selection "$DEVELOPMENT_SELECTION" "${scenes[@]}"
        ;;
    test)
        if [ ${#TEST_SCENES[@]} -ne 10 ]; then
            echo "TEST_SCENES in picasso/submit.sh must hold the ten Scannet++ test scenes"
            exit 1
        fi
        experiment test scannetpp "$SELECTION" "${TEST_SCENES[@]}"
        ;;
    contribution)
        experiment contribution_analysis replica "$SELECTION" "${VALIDATION_SCENES[@]}"
        ;;
    *)
        sed -n 2,12p "$0"
        exit 1
        ;;
esac
