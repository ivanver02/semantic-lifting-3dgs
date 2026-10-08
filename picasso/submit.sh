#!/bin/bash
# Submits each step of the evaluation, with one job per scene in the case of the steps whose scenes are independent
#   bash picasso/submit.sh development    tau phase of the two development scenes, one job each,
#                                         or of the scenes given after it
#   bash picasso/submit.sh warmup         model, masks and votes of the validation scenes, one job each,
#                                         which validation and contribution reuse, outside the analytics
#   bash picasso/submit.sh sweep          rest of the development sweep and the tau and theta selection
#   bash picasso/submit.sh validation     the seven validation scenes with the radius vote, one job each
#   bash picasso/submit.sh validation-nearest   the same grid with the nearest Gaussian as transfer operator
#   bash picasso/submit.sh selection      validation selection over both operators, run here as it only reads
#                                         the analytics, and the radius vote alone for the contribution analysis
#   bash picasso/submit.sh test           the ten Scannet++ test scenes with the selected operator, one job each
#   bash picasso/submit.sh contribution   contribution analysis of the validation scenes, one job each
#   bash picasso/submit.sh baseline-validation   the evidence per view baseline on the validation grid, one job per scene
#   bash picasso/submit.sh baseline-selection    the same rule applied to the baseline, run here
#   bash picasso/submit.sh baseline-test  the baseline at its selected point on the ten test scenes, one job each
#   bash picasso/submit.sh scannet-validation   the validation grid of both operators on the 37 Scannet++
#                                         validation scenes, one job each, for the selection inside Scannet++
#   bash picasso/submit.sh scannet-selection    the same rule over both operators on those scenes, run here
#   bash picasso/submit.sh scannet-test   the ten test scenes at the point selected on Scannet++, one job each
#   bash picasso/submit.sh analysis       2D masks against 3D results and scores for selecting Gaussians, one job
#   bash picasso/submit.sh figures        image panels of the overview and qualitative figures, one job
#   bash picasso/submit.sh report         macros, tables and figures of the manuscripts, run here as it only reads
#                                         the analytics and the analyses; the figures need matplotlib
# In the case of the steps with one job per scene, it is also possible to give scene names after the step,
# in order to submit only those scenes
# Each step is started once every job of the previous one has finished, which can be seen with squeue -p gpu_partition

source "$(dirname "$0")/env.sh"
set -eo pipefail

VALIDATION_SCENES=(office_1 office_2 office_3 office_4 room_0 room_1 room_2)
# The ten Scannet++ test scenes. They are the downloaded scenes with the most evaluated classes in their 3D
# annotation, with ties broken by scene id and leaving out the development scene, and they were chosen
# before any of them was evaluated
TEST_SCENES=(21d970d8de 27dd4da69e 09c1414f1b 0d2ee665be 25f3b7a318 3db0a1c8f3 3f15a9266d 5942004064 5eb31827b7 6115eddb86)
DEVELOPMENT_SCENES=(office_0 7831862f02)
# The Scannet++ validation scenes of the selection inside Scannet++, which only the TFG reports. They are the
# rest of the 50 downloaded scenes, leaving out the test and development ones and the two whose 3D annotation
# holds none of the target classes, 13c3e046d7 and f9f95681fd. As with the test scenes, this rule only reads
# the annotation
SCANNET_VALIDATION_SCENES=(
    1ada7a0617 286b55a2bf 31a2c91c43 3864514494 38d58a7a31 3e8bba0176 40aec5fffa 45b0dac5e3 5748ce6f01
    578511c8a9 5ee7c22ba0 5f99900f09 7b6477cb95 7bc286c1b6 825d228aec 9071e139d9 99fa5c25e1 a24f64f7fb
    a8bf42d646 a980334473 ac48a9b736 acd95847c5 b0a08200c9 bcd2436daf bde1e479ad c49a8c6cff c4c04e6d6c
    c50d2d1d42 c5439f4607 cc5237fd77 d755b3d9d8 e398684d27 e7af285f7d f2dc06b1d2 f3685d06a9 f3d64c30f8
    fb5a96b1a2
)
BETAS=(0.50 0.70 0.90 0.94 0.95 0.96 0.97 0.975 0.98 0.985 0.99 0.995 0.999)
TAU_GRID=(0.02 0.03 0.05 0.08 0.10 0.15 0.20)
THETA_GRID=(0.3 0.4 0.5 0.6 0.7)

ANALYTICS=$TFG_DATA/analytics
DEVELOPMENT_SELECTION=$ANALYTICS/tau_theta_selection.json
# The selection over both transfer operators, which is the one that the test reads, and the one over the
# radius vote alone, from which the contribution analysis varies one factor at a time
SELECTION=$ANALYTICS/selection_transfer.json
RADIUS_SELECTION=$ANALYTICS/selection.json
# The selection of the baseline, which thresholds E+ per view and keeps the rest of the method as it is
BASELINE_SELECTION=$ANALYTICS/selection_baseline.json
# The selection over both operators on the Scannet++ validation scenes, which the scannet-test step reads
SCANNET_SELECTION=$ANALYTICS/selection_scannet.json

DEVELOPMENT_SWEEP=(
    evaluation.scripts.development_sweep
    --replica-data-root "$TFG_DATA/replica" --scannetpp-data-root "$TFG_DATA/scannetpp"
    --replica-scene "${DEVELOPMENT_SCENES[0]}" --scannetpp-scene "${DEVELOPMENT_SCENES[1]}"
    --tau-grid "${TAU_GRID[@]}" --theta-grid "${THETA_GRID[@]}"
)

# The jobs write their logs relative to the directory from which they are submitted
mkdir -p "$TFG_ROOT/logs"
cd "$TFG_ROOT"

submit() {
    # Submits one job with the name of its step and scene, so that its log is logs/<name>-<job id>.out
    local name=$1
    shift
    sbatch --job-name="$name" "$TFG_REPO/picasso/job.sbatch" "$@"
}

experiment() {
    # Submits one job per scene of an experiment, each one with the full list of scenes and its own --only-scene.
    # When ONLY holds scenes, only those are submitted, in order to continue the ones that did not finish.
    # The jobs take the name of the step, and EXPERIMENT_ARGS holds the extra arguments of the experiment
    local step=$1 name=$2 dataset=$3 selection=$4
    shift 4
    if [ ! -e "$selection" ]; then
        echo "missing $selection, the previous step has not finished"
        exit 1
    fi
    local scenes=() scene
    for scene in "$@"; do
        scenes+=(--scene "$scene")
    done
    for scene in "${ONLY[@]}"; do
        if [[ " $* " != *" $scene "* ]]; then
            echo "$scene is not a scene of the $step step"
            exit 1
        fi
    done
    for scene in "$@"; do
        if [ ${#ONLY[@]} -gt 0 ] && [[ " ${ONLY[*]} " != *" $scene "* ]]; then
            continue
        fi
        submit "$step-$scene" evaluation.scripts.experiment --experiment "$name" --selection "$selection" \
            --data-root "$TFG_DATA/$dataset" --output-root "$TFG_DATA/$dataset/eval" \
            "${scenes[@]}" --only-scene "$scene" "${EXPERIMENT_ARGS[@]}"
    done
}

# The scenes given after the name of the step limit that step to them
ONLY=("${@:2}")
EXPERIMENT_ARGS=()

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
        # The votes do not depend on tau, theta or gamma. In this way, this run with the default configuration
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
        experiment validation validation replica "$DEVELOPMENT_SELECTION" "${VALIDATION_SCENES[@]}"
        ;;
    validation-nearest)
        # The votes and the selected Gaussians are already cached, so these units only transfer and score again
        EXPERIMENT_ARGS=(--transfer nearest_neighbor_label)
        experiment validation-nearest validation replica "$DEVELOPMENT_SELECTION" "${VALIDATION_SCENES[@]}"
        ;;
    selection)
        scenes=()
        for scene in "${VALIDATION_SCENES[@]}"; do
            scenes+=(--scene "$scene")
        done
        cd "$TFG_REPO"
        "$TFG_PYTHON" -m evaluation.scripts.selection --analytics "$ANALYTICS" --output "$SELECTION" \
            --development-selection "$DEVELOPMENT_SELECTION" "${scenes[@]}"
        "$TFG_PYTHON" -m evaluation.scripts.selection --analytics "$ANALYTICS" --output "$RADIUS_SELECTION" \
            --development-selection "$DEVELOPMENT_SELECTION" "${scenes[@]}" --transfer radius_vote > /dev/null
        ;;
    test)
        if [ ${#TEST_SCENES[@]} -ne 10 ]; then
            echo "TEST_SCENES in picasso/submit.sh must hold the ten Scannet++ test scenes"
            exit 1
        fi
        experiment test test scannetpp "$SELECTION" "${TEST_SCENES[@]}"
        ;;
    contribution)
        experiment contribution_analysis contribution_analysis replica "$RADIUS_SELECTION" "${VALIDATION_SCENES[@]}"
        ;;
    baseline-validation)
        # The votes, the masks and the models are already cached, so these units only threshold, transfer and score
        experiment baseline-validation baseline_validation replica "$DEVELOPMENT_SELECTION" "${VALIDATION_SCENES[@]}"
        ;;
    baseline-selection)
        scenes=()
        for scene in "${VALIDATION_SCENES[@]}"; do
            scenes+=(--scene "$scene")
        done
        cd "$TFG_REPO"
        "$TFG_PYTHON" -m evaluation.scripts.selection --analytics "$ANALYTICS" --output "$BASELINE_SELECTION" \
            --development-selection "$DEVELOPMENT_SELECTION" "${scenes[@]}" --baseline
        ;;
    baseline-test)
        experiment baseline-test baseline_test scannetpp "$BASELINE_SELECTION" "${TEST_SCENES[@]}"
        ;;
    scannet-validation)
        # The same tau and theta of the development sweep, so only the operator, beta and gamma are selected again.
        # Each job runs both operators, so a scene has every result once its job finishes. Since the quota of files
        # of fscratch does not hold the 37 scenes at once, they are submitted in batches, giving their names
        experiment scannet-validation scannet_validation scannetpp "$DEVELOPMENT_SELECTION" \
            "${SCANNET_VALIDATION_SCENES[@]}"
        ;;
    scannet-selection)
        scenes=()
        for scene in "${SCANNET_VALIDATION_SCENES[@]}"; do
            scenes+=(--scene "$scene")
        done
        cd "$TFG_REPO"
        "$TFG_PYTHON" -m evaluation.scripts.selection --analytics "$ANALYTICS" --output "$SCANNET_SELECTION" \
            --development-selection "$DEVELOPMENT_SELECTION" "${scenes[@]}" --scannet
        ;;
    scannet-test)
        experiment scannet-test scannet_test scannetpp "$SCANNET_SELECTION" "${TEST_SCENES[@]}"
        ;;
    analysis)
        sbatch --job-name=analysis "$TFG_REPO/picasso/analysis.sbatch"
        ;;
    figures)
        submit figures evaluation.scripts.figure_renders --data-root "$TFG_DATA"
        ;;
    report)
        cd "$TFG_REPO"
        report=(--analytics "$ANALYTICS" --analysis "$TFG_DATA/analysis")
        "$TFG_PYTHON" -m evaluation.scripts.make_macros "${report[@]}" --output "$TFG_DATA/report/macros_measured.tex"
        "$TFG_PYTHON" -m evaluation.scripts.make_tables "${report[@]}" --out "$TFG_DATA/report/tables"
        "$TFG_PYTHON" -m evaluation.scripts.make_figures "${report[@]}" --out "$TFG_DATA/report/figures" \
            --renders "$TFG_DATA/scannetpp/figures"
        ;;
    *)
        sed -n 2,27p "$0"
        exit 1
        ;;
esac
