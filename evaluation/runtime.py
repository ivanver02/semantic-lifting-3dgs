# Small wrappers for running the project containers

import os
import json
import subprocess
import uuid
from pathlib import Path


class Runtime:
    """ Run training, lifting scripts and COLMAP through Docker """

    def __init__(self, repo_root, data_root, train_image="tfgivanverdugo/semantic-fusion-gs-train:cuda11.6",
                 lifting_image="tfgivanverdugo/semantic-fusion-fusion:cuda11.6",
                 colmap_image="tfgivanverdugo/semantic-fusion-colmap:3.13.0-cpu"):

        """ Store host roots and the three container image names """
        self.repo_root = Path(repo_root).resolve()
        self.data_root = Path(data_root).resolve()
        self.train_image = train_image
        self.lifting_image = lifting_image
        self.colmap_image = colmap_image
        self._stage_peak_cuda_memory_bytes = None
        self._stage_peak_cuda_memory_reserved_bytes = None

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

        gpu dds the NVIDIA runtime when enabled.
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

            # Set writable cache locations because the repository mount is read only
        result += [
            "-e", "HOME=/tmp",
            "-e", "MPLCONFIGDIR=/tmp/matplotlib",
            "-e", "YOLO_CONFIG_DIR=/tmp/Ultralytics",
            "-e", "QT_QPA_PLATFORM=offscreen",
            "-v", f"{self.repo_root}:/repo:ro",
            "-v", f"{self.data_root}:/data:rw",
            "-w", "/repo",

    # Add Docker image and working directory
            image,
        ]

        # Append the program and its arguments after all Docker options
        return result + list(command)

    def _run(self, command):
        """ Run a prepared command and raise errors from failed stages """

        # Print the command so a failed stage can be reproduced manually
        print("Docker command: ", " ".join(str(item) for item in command))
        subprocess.run(command, check=True, text=True, cwd=str(self.repo_root))

    def begin_stage(self):
        """ Reset the maximum CUDA value collected from container processes """
        self._stage_peak_cuda_memory_bytes = None
        self._stage_peak_cuda_memory_reserved_bytes = None

    def end_stage(self):
        """ Return and clear the maximum CUDA value collected for one stage """
        peak = {
            "allocated": self._stage_peak_cuda_memory_bytes,
            "reserved": self._stage_peak_cuda_memory_reserved_bytes,
        }
        self._stage_peak_cuda_memory_bytes = None
        self._stage_peak_cuda_memory_reserved_bytes = None
        return peak

    def _run_python(self, image, gpu, target_kind, target, arguments):
        """ Run a Python target through the memory wrapper in the container """
        metrics_dir = self.data_root / ".evaluation_runtime_metrics"
        metrics_path = metrics_dir / f"{uuid.uuid4().hex}.json"
        mapped_metrics_path = self._container_path(str(metrics_path))
        args = [self._container_path(str(item)) for item in arguments]
        command = [
            "python", "evaluation/runtime_metrics.py",
            f"--{target_kind}", target,

    # Finish the memory metrics command
            "--metrics-path", mapped_metrics_path,
            "--", *args,
        ]

        # Run the target and merge measured peaks
        try:
            # Never allow a previous interrupted container to provide this run's
            metrics_path.unlink(missing_ok=True)
            self._run(self._docker_command(image, gpu, command))
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            for key, attribute in (
                ("peak_cuda_memory_bytes", "_stage_peak_cuda_memory_bytes"),
                ("peak_cuda_memory_reserved_bytes", "_stage_peak_cuda_memory_reserved_bytes"),
            ):

                # Keep the largest value for the stage
                peak = metrics.get(key)
                if peak is not None:
                    peak = int(peak)
                    previous = getattr(self, attribute)
                    if previous is None or peak > previous:
                        setattr(self, attribute, peak)
        finally:
            metrics_path.unlink(missing_ok=True)

    def run_lifting(self, script, arguments):
        """ Run a repository Python script in the lifting container """
        self._run_python(self.lifting_image, True, "script", script, arguments)

    def run_lifting_module(self, module, arguments):
        """ Run a Python module in the lifting container """
        # Module execution keeps relative imports working inside the repository
        self._run_python(self.lifting_image, True, "module", module, arguments)

    def run_train(self, dataset_dir, model_dir, iterations, resolution, data_device="cuda"):
        """
        Run Gaussian training with the selected data and image settings

        resolution is given to the training script as -r. Values such as 1 and 2 select the original or half image resolution.
        """

        # Build the training arguments
        arguments = [
            "-s", str(dataset_dir),
            "-m", str(model_dir),
            "-r", str(resolution),
            "--iterations", str(iterations),
            "--save_iterations", str(iterations),
            "--checkpoint_iterations", str(iterations),
            "--data_device", data_device,
        ]

        # Only the dataset and model arguments are mounted paths in this list
        mapped = [self._container_path(item) if index in (1, 3) else item for index, item in enumerate(arguments)]
        self._run_python(self.train_image, True, "script", "train.py", mapped)

    def run_colmap(self, arguments):
        """ Run COLMAP in the CPU container with the supplied arguments """
        # COLMAP receives all input and output paths through the shared mounts
        args = list(arguments)
        mapped = [self._container_path(str(item)) for item in args]
        command = self._docker_command(self.colmap_image, False, ["colmap"] + mapped)
        self._run(command)