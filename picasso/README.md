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

`picasso/submit.sh` submits each step of the evaluation, with one job per
scene wherever the scenes are independent, and `picasso/job.sbatch` runs each
job on one A100. The log of a job is `logs/<step>-<scene>-<job id>.out`, and a
step starts once every job of the previous one has finished:

```bash
cd $FSCRATCH/tfg
bash repo/picasso/submit.sh development    # tau phase of office_0 and 7831862f02
bash repo/picasso/submit.sh sweep          # theta phase and tau_theta_selection.json
bash repo/picasso/submit.sh validation     # the seven validation scenes
bash repo/picasso/submit.sh selection      # selection.json, on the login node
bash repo/picasso/submit.sh test           # the ten Scannet++ test scenes
bash repo/picasso/submit.sh contribution   # the contribution analysis
```

The test scenes go in `TEST_SCENES` of `picasso/submit.sh`. The tables, figures
and macros only read the analytics directory, so it can be copied to another
machine to run them.

Each job asks for two hours in the `short` QoS, because SLURM fits such jobs in
the gaps between larger ones and they start within minutes. Ten minutes before
the end, the run stops between stages, or training saves a resume checkpoint,
and the job submits the same command again as a new part. The new part skips the
cached stages and continues the training from the checkpoint. A chain stops
after `TFG_MAX_PARTS` parts (12 by default), and an error never submits a new
part. A stopped chain continues by submitting the same step again, which skips
the scenes that already have results.

## Notes

- Once a Scannet++ scene has its undistorted COLMAP model, the job removes
  `dslr/resized_images`, as nothing reads it again and fscratch has a file quota.
- Parallel scenes append to the same CSV tables, which is safe because
  `AnalyticsStore` locks the store for every write.
- A run is recorded in the analytics when it finishes, with the stages of its
  last part: the ones cached by earlier parts appear as cache hits.
- `squeue -u $USER` lists the jobs, and `scancel <job id>` stops one.
