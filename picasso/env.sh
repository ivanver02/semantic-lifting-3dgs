# Settings shared by the Picasso scripts, sourced by build_images.sh and job.sbatch
# Each variable can be exported beforehand to override its default

# Everything lives under $FSCRATCH/tfg: the repository, the images, the data and the logs
export TFG_ROOT=${TFG_ROOT:-$FSCRATCH/tfg}
export TFG_REPO=${TFG_REPO:-$TFG_ROOT/repo}
export TFG_DATA=${TFG_DATA:-$TFG_ROOT/data}
export TFG_SIF_DIR=${TFG_SIF_DIR:-$TFG_ROOT/sifs}

# Picasso has no Docker, so evaluation.runtime runs every stage in the .sif images
export TFG_RUNTIME=apptainer

# The launcher runs in the cluster conda base, with opencv-python-headless and plyfile installed with pip --user
export TFG_PYTHON=${TFG_PYTHON:-/mnt/home/soft/miniconda/programs/x86_64/miniconda3_py310_23.1.0/bin/python}

module load apptainer
