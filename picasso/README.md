# Running on Picasso

As most clusters, Picasso manages the jobs with SLURM, but Docker is not available on it.
For this reason, the same three images are built with Apptainer from
`containers/apptainer/*.def`, and `evaluation.runtime` runs every stage with
`apptainer exec` when `TFG_RUNTIME=apptainer`. Without that variable, everything
keeps running with Docker as before.

All the files of the project are kept under `$FSCRATCH/tfg`, with the following structure:

```
$FSCRATCH/tfg/
  repo/            this repository, with yolo26x-seg.pt at its root
  sifs/            colmap.sif, lifting.sif and gs-train.sif
  data/replica/    one directory per scene (office_0 ... room_2)
  data/scannetpp/  metadata/semantic_classes.txt and validation_data/<scene>
  data/analytics/  CSV tables written by the runs, shared by both datasets
  data/analysis/   CSV tables of the two analyses that read masks and votes
  data/report/     macros, tables and figures of the manuscripts
  logs/            build and job logs
```

Note that `evaluation.run` itself does not run inside the images. It runs in the conda
base of the cluster, and it only needs a few extra packages on top of the numpy and
scipy that are already installed there.

## Setup (only the first time)

```bash
cd $FSCRATCH/tfg/repo
git submodule update --init --recursive submodules/diff-gaussian-rasterization submodules/simple-knn submodules/fused-ssim
source picasso/env.sh
$TFG_PYTHON -m pip install --user opencv-python-headless plyfile
nohup bash picasso/build_images.sh > $FSCRATCH/tfg/logs/build_all.log 2>&1 &
```

The images are built on the login node. This is due to the fact that the compute
nodes have no internet access. The same happens at run time, so the YOLO weights
must already be in the root of the repository before submitting any job.

## Jobs

The evaluation is divided into steps, and `picasso/submit.sh` submits each of them,
with one job per scene in the case of the steps whose scenes are independent. Each
job is run by `picasso/job.sbatch` on one A100, and its log is written to
`logs/<step>-<scene>-<job id>.out`. The steps are the following ones, in this order,
and each step is started once every job of the previous one has finished:

```bash
cd $FSCRATCH/tfg
bash repo/picasso/submit.sh development    # tau phase of office_0 and 7831862f02
bash repo/picasso/submit.sh warmup         # models, masks and votes of the validation scenes
bash repo/picasso/submit.sh sweep          # theta phase and tau_theta_selection.json
bash repo/picasso/submit.sh validation     # the seven validation scenes, radius vote
bash repo/picasso/submit.sh validation-nearest   # the same grid, nearest Gaussian
bash repo/picasso/submit.sh selection      # selection_transfer.json and selection.json, on the login node
bash repo/picasso/submit.sh test           # the ten Scannet++ test scenes, selected operator
bash repo/picasso/submit.sh contribution   # the contribution analysis
bash repo/picasso/submit.sh baseline-validation   # the evidence per view baseline on the validation grid
bash repo/picasso/submit.sh baseline-selection    # selection_baseline.json, on the login node
bash repo/picasso/submit.sh baseline-test         # the baseline at its selected point on the test scenes
bash repo/picasso/submit.sh analysis       # 2D masks against 3D results, scores for selecting Gaussians
bash repo/picasso/submit.sh figures        # image panels of the overview and qualitative figures
bash repo/picasso/submit.sh report         # macros, tables and figures, on the login node
```

The `warmup` step is optional. It runs the validation scenes with the default configuration
and without recording anything, so that the stages that do not depend on the selection are
already cached while the development sweep runs. In addition, `development office_0` submits
a single development scene, and the test scenes are the ones listed in `TEST_SCENES` of
`picasso/submit.sh`.

In the validation, the two transfer operators from Gaussians to mesh are compared as one
more choice of the operating point. That is, `selection` applies the rule over the operator,
beta and gamma together and writes `selection_transfer.json`, which is the one that the test
reads. It also applies the same rule to the radius vote alone and writes `selection.json`,
from which the contribution analysis varies one factor at a time. Each operator writes its
runs under its own variant, `frozen_g*` or `frozen_nearest_g*`, so that neither of them
replaces the results of the other.

The baseline thresholds the target evidence per view instead of the evidence fraction, and it
keeps the rest of the configuration of the method, with the nearest Gaussian as transfer
operator. Since it reads the cached votes, its units only threshold, transfer and score.
Regarding its outputs, the runs are written under the variants `baseline_per_view_nearest_g*`,
and the selections next to the votes, in directories that end in `_per_view`. In this way,
nothing of the method is replaced.

`analysis` reads the masks and the cached votes, so it runs in `lifting.sif` as a single job,
and it writes `mask_agreement.csv` and `threshold_scores.csv` to `data/analysis`. Finally, the
macros, tables and figures only read `data/analytics` and `data/analysis`. For this reason,
both directories can be copied to another machine in order to generate them there:

```bash
python -m evaluation.scripts.make_macros --analytics analytics --analysis analysis --output report/macros_measured.tex
python -m evaluation.scripts.make_tables --analytics analytics --analysis analysis --out report/tables
python -m evaluation.scripts.make_figures --analytics analytics --analysis analysis --out report/figures
```

It should be noted that the figures need matplotlib, which may not be installed in the conda
base of the cluster.

Each job asks for two hours in the `short` QoS. The reason for this is that SLURM fits these
short jobs in the gaps left by the larger ones, and in most cases they start within minutes.
Ten minutes before the end of the job, the run stops between two stages or, in the case of
the training, it saves a resume checkpoint. Then, the job submits the same command again as a
new part, which skips the cached stages and continues the training from the checkpoint. A
chain of parts stops after `TFG_MAX_PARTS` parts (12 by default), and a new part is never
submitted after an error. As for the resources, the new part takes its time and QoS from the
directives of `job.sbatch`, and its memory from `SBATCH_MEM_PER_NODE` in the case that the
first part was submitted with it. In order to continue a chain that has stopped, it is
necessary to submit the same step again, which skips the scenes that already have results,
or to submit only the scenes that did not finish, as in `submit.sh test 09c1414f1b 3f15a9266d`.

## Notes

- Once a Scannet++ scene has its undistorted COLMAP model, the job removes
  `dslr/resized_images`. This is due to the fact that nothing reads these images again,
  and fscratch has a quota on the number of files.
- The scenes that run in parallel append their rows to the same CSV tables. This is safe,
  because `AnalyticsStore` locks the store for every write.
- A run is recorded in the analytics when it finishes, with the stages of its last part.
  That is, the stages cached by the earlier parts appear as cache hits.
- `squeue -p gpu_partition` lists the jobs, and `scancel <job id>` stops one of them. Note
  that the `squeue` of Picasso is a wrapper that does not handle `-o` reliably. For this
  reason, the jobs never query it, and the state and exit code of the finished jobs are
  obtained with `sacct`.
