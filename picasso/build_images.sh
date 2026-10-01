#!/bin/bash
# Builds the Apptainer images on the login node, which is the only one with internet access
#   bash picasso/build_images.sh                 builds colmap, lifting and gs-train
#   bash picasso/build_images.sh lifting         builds only the given images
# The CUDA images take a long time to build, so it is better to leave it running in the background:
#   nohup bash picasso/build_images.sh > $FSCRATCH/tfg/logs/build_all.log 2>&1 &

source "$(dirname "$0")/env.sh"
set -eo pipefail

# The build unpacks each image into many files. For this reason, it works in the local /tmp instead of fscratch,
# which has a quota on the number of files
export APPTAINER_TMPDIR=${APPTAINER_TMPDIR:-/tmp/$USER-apptainer}
mkdir -p "$APPTAINER_TMPDIR" "$TFG_SIF_DIR" "$TFG_ROOT/logs"

# The paths in the %files section of the recipes are relative to the root of the repository
cd "$TFG_REPO"
images=("$@")
if [ ${#images[@]} -eq 0 ]; then
    images=(colmap lifting gs-train)
fi

# The training image compiles the three CUDA submodules, so it is necessary to check them out first
if [[ " ${images[*]} " == *" gs-train "* ]] && [ ! -e submodules/simple-knn/setup.py ]; then
    echo "missing submodules, run: git submodule update --init --recursive submodules/diff-gaussian-rasterization submodules/simple-knn submodules/fused-ssim"
    exit 1
fi

status=0
for image in "${images[@]}"; do
    echo "building $image"
    if ! apptainer build --fakeroot --force "$TFG_SIF_DIR/$image.sif" "containers/apptainer/$image.def" \
            > "$TFG_ROOT/logs/build_$image.log" 2>&1; then
        echo "FAILED $image, see $TFG_ROOT/logs/build_$image.log"
        status=1
    fi
done

# Once the .sif files exist, the layer cache is not needed anymore
apptainer cache clean -f
exit $status
