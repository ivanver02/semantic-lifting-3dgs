# Running on Picasso

Picasso has no Docker, so the same three images are built with Apptainer from
`containers/apptainer/*.def`, and `evaluation.runtime` runs every stage with
`apptainer exec` when `TFG_RUNTIME=apptainer`. Without that variable everything
keeps running with Docker as before.

Everything lives under `$FSCRATCH/tfg`:

```
$FSCRATCH/tfg/
  repo/            this repository, with yolo26x-seg.pt at its root
  sifs/            colmap.sif, lifting.sif and gs-train.sif
  data/replica/    one directory per scene (office_0 ... room_2)
  data/scannetpp/  metadata/semantic_classes.txt and validation_data/<scene>
  data/analytics/  CSV tables written by the runs, shared by both datasets
  logs/            build and job logs
```

`evaluation.run` itself runs outside the images, in the cluster conda base, and
only needs a few extra packages on top of its numpy and scipy.

## Once

```bash
cd $FSCRATCH/tfg/repo
git submodule update --init --recursive submodules/diff-gaussian-rasterization submodules/simple-knn submodules/fused-ssim
source picasso/env.sh
$TFG_PYTHON -m pip install --user opencv-python-headless plyfile
nohup bash picasso/build_images.sh > $FSCRATCH/tfg/logs/build_all.log 2>&1 &
```

The images are built on the login node because the compute nodes have no
internet access. They also have no internet access at run time, so the YOLO
weights must already be in the repository root.

## Jobs

Jobs are submitted from `$FSCRATCH/tfg`, after `source repo/picasso/env.sh` so
that `$TFG_DATA` is defined. `picasso/job.sbatch` runs any Python module of the
repository on one A100, and its log is written to `logs/tfg-<job id>.out`.

Development sweep, one job, since the theta phase depends on the tau selected
from both development scenes:

```bash
sbatch repo/picasso/job.sbatch evaluation.scripts.development_sweep \
    --replica-data-root $TFG_DATA/replica --scannetpp-data-root $TFG_DATA/scannetpp \
    --scannetpp-scene 7831862f02 \
    --tau-grid 0.02 0.03 0.05 0.08 0.10 --theta-grid 0.3 0.4 0.5 0.6 0.7
```

Experiments, as an array with one task per scene. The index of each task picks
its scene from the `--scene` list, which must still hold every scene:

```bash
sbatch --array=0-6 repo/picasso/job.sbatch evaluation.scripts.experiment \
    --experiment validation --selection $TFG_DATA/analytics/tau_theta_selection.json \
    --data-root $TFG_DATA/replica --output-root $TFG_DATA/replica/eval \
    --scene office_1 --scene office_2 --scene office_3 --scene office_4 \
    --scene room_0 --scene room_1 --scene room_2
```

The selection only reads the CSV tables, so it runs on the login node:

```bash
cd repo && $TFG_PYTHON -m evaluation.scripts.selection \
    --analytics $TFG_DATA/analytics --output $TFG_DATA/analytics/selection.json \
    --development-selection $TFG_DATA/analytics/tau_theta_selection.json \
    --scene office_1 --scene office_2 --scene office_3 --scene office_4 \
    --scene room_0 --scene room_1 --scene room_2
```

The test (`--array=0-9`, Scannet++ roots and its ten scenes) and the
contribution analysis (`--array=0-6`) take `--selection $TFG_DATA/analytics/selection.json`.
The tables, figures and macros only read the analytics directory, so it can be
copied to another machine to run them.

## Notes

- Once a Scannet++ scene has its undistorted COLMAP model, the job removes
  `dslr/resized_images`, as nothing reads it again and fscratch has a file quota.
- Parallel scenes append to the same CSV tables, which is safe because
  `AnalyticsStore` locks the store for every write.
- `squeue -u $USER` lists the jobs, and `scancel <job id>` stops one.
