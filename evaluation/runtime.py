# Small wrappers for running the project containers

import os
import json
import subprocess
import uuid
from pathlib import Path


TRAIN_IMAGE = "tfgivanverdugo/semantic-fusion-gs-train:cuda11.6"
LIFTING_IMAGE = "tfgivanverdugo/semantic-fusion-fusion:cuda11.6"
COLMAP_IMAGE = "tfgivanverdugo/semantic-fusion-colmap:3.13.0-cpu"


class Runtime:
    """ Run training, lifting scripts and COLMAP through Docker """

    def __init__(self, repo_root, data_root):
        """ Store host roots and the peak CUDA memory of the current stage """
        self.repo_root = Path(repo_root).resolve()
        self.data_root = Path(data_root).resolve()
        self._stage_peak_cuda_memory_bytes = None

    def _container_path(self, value):
        """ Convert a host path into its mounted container path """

        # Leave relative arguments and strings not related to paths as is
        path = Path(value)
        if not path.is_absolute():
            return value
        try:
            # Repository files are mounted as read only at /repo
            return "/repo/" + path.relative_to(self.repo_root).as_posix()
        except ValueError:
            pass
        try:
            # Dataset and output files are mounted as read and write at /data
            return "/data/" + path.relative_to(self.data_root).as_posix()
        except ValueError:
            return value

    def _docker_command(self, image, gpu, command):
        """
        Build a Docker command from an image and its command arguments

        gpu adds the NVIDIA runtime when enabled.
        command is the sequence of program arguments that runs inside the container.
        """

        # Start a temporary container that is removed after the command exits
        result = ["docker", "run", "--rm"]
        if gpu:
            # Training and rasterization need access to all visible NVIDIA GPUs
            result += ["--gpus", "all"]
        if hasattr(os, "getuid"):
            # Keep files created in mounted directories owned by the host user
            result += ["--user", f"{os.getuid()}:{os.getgid()}"]

        # Writable cache locations because the repository mount is read only
        result += [
            "-e", "HOME=/tmp",
            "-e", "MPLCONFIGDIR=/tmp/matplotlib",
            "-e", "YOLO_CONFIG_DIR=/tmp/Ultralytics",
            "-e", "QT_QPA_PLATFORM=offscreen",
            "-v", f"{self.repo_root}:/repo:ro",
            "-v", f"{self.data_root}:/data:rw",
            "-w", "/repo",
            image,
        ]

        # Append the program and its arguments after all Docker options, with host paths mapped
        return result + [self._container_path(str(item)) for item in command]

    def _run(self, command):
        """ Run a prepared command and raise errors from failed stages """

        # Print the command so a failed stage can be reproduced manually
        print("Docker command: ", " ".join(str(item) for item in command))
        subprocess.run(command, check=True, text=True, cwd=str(self.repo_root))

    def end_stage(self):
        """ Return and clear the maximum CUDA value collected for one stage """
        peak = self._stage_peak_cuda_memory_bytes
        self._stage_peak_cuda_memory_bytes = None
        return peak

    def _run_python(self, image, target_kind, target, arguments):
        """ Run a Python target through the memory wrapper in the container """
        metrics_path = self.data_root / ".evaluation_runtime_metrics" / f"{uuid.uuid4().hex}.json"
        command = [
            "python", "evaluation/runtime_metrics.py",
            f"--{target_kind}", target,
            "--metrics-path", metrics_path,
            "--", *arguments,
        ]

        # Run the target and keep the largest measured peak for the stage
        try:
            self._run(self._docker_command(image, True, command))
            peak = json.loads(metrics_path.read_text(encoding="utf-8"))["peak_cuda_memory_bytes"]
            if peak is not None:
                self._stage_peak_cuda_memory_bytes = max(peak, self._stage_peak_cuda_memory_bytes or 0)
        finally:
            metrics_path.unlink(missing_ok=True)

    def run_lifting(self, script, arguments):
        """ Run a repository Python script in the lifting container """
        self._run_python(LIFTING_IMAGE, "script", script, arguments)

    def run_lifting_module(self, module, arguments):
        """ Run a Python module in the lifting container """
        # Module execution keeps relative imports working inside the repository
        self._run_python(LIFTING_IMAGE, "module", module, arguments)

    def run_train(self, dataset_dir, model_dir, iterations, resolution, data_device="cuda"):
        """
        Run Gaussian training with the selected data and image settings

        resolution is given to the training script as -r. Values such as 1 and 2 select the original or half image resolution.
        """
        self._run_python(TRAIN_IMAGE, "script", "train.py", [
            "-s", dataset_dir,
            "-m", model_dir,
            "-r", resolution,
            "--iterations", iterations,
            "--save_iterations", iterations,
            "--checkpoint_iterations", iterations,
            "--data_device", data_device,
        ])

    def run_colmap(self, arguments):
        """ Run COLMAP in the CPU container with the supplied arguments """
        # COLMAP receives all input and output paths through the shared mounts
        self._run(self._docker_command(COLMAP_IMAGE, False, ["colmap", *arguments]))
