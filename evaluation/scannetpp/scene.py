# Scannet++ scene loading, taxonomy conversion, COLMAP preparation and GT masks

import shutil
from pathlib import Path
from plyfile import PlyData

import numpy as np

from ..common import SceneData, TargetClassInfo


CLASSES = [
    TargetClassInfo("bench", "bench", 14),
    TargetClassInfo("chair", "chair", 57),
    TargetClassInfo("table", "dining table", 61),
    TargetClassInfo("tv", "tv", 63),
    TargetClassInfo("laptop", "laptop", 64),
    TargetClassInfo("sink", "sink", 72),
    TargetClassInfo("clock", "clock", 75),
]

DATASET_LABELS = {
    "bench": {"bench", "experiment bench", "laboratory bench", "work bench",
               "window bench", "wood bench"},
    "chair": {
        "chair", "office chair", "armchair", "arm chair", "dining chair",
        "folding chair", "office visitor chair", "rolling chair", "lounge chair",
        "sofa chair", "deck chair", "papasan chair",
    },
    "table": {"table", "dining table", "office table", "conference table",
              "joined tables"},
    "tv": {"tv", "television", "tv screen"},
    "laptop": {"laptop"},
    "sink": {"sink", "kitchen sink", "bathroom sink", "washbasin", "wash basin"},
    "clock": {"clock", "wall clock", "table clock", "alarm clock"},
}


class ScannetScene:
    """ Load Scannet++ data and convert it to the common evaluation format """

    def __init__(self, data_root, scene, support_dir):
        """
        Store the scene paths and the directory containing generated GT

        - data_root: the root directory of the Scannet++ dataset
        - scene: the name of the scene to process
        - support_dir: the directory containing rasterized masks and visible vertices
        """
        self.data_root = Path(data_root)
        self.scene = scene
        self.scene_root = self.data_root / "validation_data" / scene
        self.scans = self.scene_root / "scans"
        self.support_dir = Path(support_dir)

    @property
    def metadata_path(self):
        """ Return the semantic class metadata file for Scannet++ """
        # Each line position in this file is the dataset semantic ID used by the mesh labels
        return self.data_root / "metadata" / "semantic_classes.txt"

    @property
    def prepared_dir(self):
        """ Return the directory containing the prepared COLMAP model """
        return self.scene_root / "dslr" / "undistorted_colmap"

    def _load_mesh(self):
        """ Load vertex positions and Scannet++ dataset IDs """

        # Scannet++ stores one semantic dataset ID per mesh vertex, unlike Replica which stores labels on faces
        ply = PlyData.read(str(self.scans / "mesh_aligned_0.05_semantic.ply"))
        vertex = ply["vertex"]

        # Load vertex coordinates
        vertices = np.vstack([vertex["x"], vertex["y"], vertex["z"]]).T.astype(np.float64)

        # Load the Scannet++ dataset ID assigned to each vertex
        labels = np.asarray(vertex["label"], dtype=np.int64)
        return vertices, labels

    def _dataset_ids_to_local_ids(self, names):
        """ Map Scannet++ dataset IDs to SceneData local IDs """

        # Convert Scannet++ dataset names into the local class ordering used by metrics
        mapping = {}
        for local_id, item in enumerate(CLASSES):

            # A main class can correspond to several Scannet++ names
            for name in DATASET_LABELS[item.name]:
                if name in names:
                    mapping[names.index(name)] = local_id
        return mapping

    def load_data(self):
        """ Load mesh labels and generated visibility as common scene data """

        # Load Scannet++ dataset IDs before converting them to local IDs
        vertices, dataset_labels = self._load_mesh()

        # The metadata order defines which integer ID corresponds to each Scannet++ class name
        dataset_names = [line.strip().lower() for line in self.metadata_path.read_text().splitlines()]
        dataset_ids_to_local_ids = self._dataset_ids_to_local_ids(dataset_names)

        # Keep unknown dataset IDs invalid
        semantic = np.asarray([dataset_ids_to_local_ids.get(int(label), -1) for label in dataset_labels], dtype=np.int64)

        # The GT mask stage records the vertices observed by the rendered camera views, which indicates which vertices are visible
        visible = np.load(self.support_dir / "support.npz")["visible_vertices"].astype(bool)
        if visible.shape != (len(vertices),):
            raise ValueError("Scannet++ GT support and semantic mesh use different vertex counts")

        return SceneData(
            dataset="scannetpp",
            scene=self.scene,
            vertices=vertices,
            semantic_labels=semantic,
            annotated=((dataset_labels >= 0) & (dataset_labels < len(dataset_names))),
            visible=visible,
            classes=CLASSES,
        )

    def prepare_dataset(self, runtime, max_image_size=1600):
        """
        Prepare Scannet++ DSLR (which have distortion, and we want undistorted ones) images and create a COLMAP model
        """
        output = self.prepared_dir

        # Prefer the dataset resized images, but if not available, use the original images
        images = self.scene_root / "dslr" / "resized_images"
        if not images.exists():
            images = self.scene_root / "dslr" / "images"

        # Undistort the images and write an output directory ready for COLMAP using the existing COLMAP reconstruction
        runtime.run_colmap([
            "image_undistorter",
            "--image_path", str(images),
            "--input_path", str(self.scene_root / "dslr" / "colmap"),
            "--output_path", str(output),
            "--output_type", "COLMAP",
            "--max_image_size", str(max_image_size),
        ])

        # Normalize COLMAP sparse output so we can always use sparse/0
        sparse = output / "sparse"
        sparse_zero = sparse / "0"
        if sparse.exists() and not sparse_zero.exists():

            # COLMAP can place its files directly in sparse, but the rest of the project expects sparse/0
            sparse_zero.mkdir(parents=True, exist_ok=True)
            for item in sparse.iterdir():
                if item.is_file() and item.suffix in {".bin", ".txt"}:
                    shutil.move(str(item), str(sparse_zero / item.name))  # shutil.move can move across filesystems

    def generate_gt_masks(self, runtime, output_dir, bands=4):
        """
        Generate rasterized Scannet++ GT masks and visibility support

        - bands: number of horizontal image bands used to reduce GPU memory
        """

        # Rasterize the mesh inside the lifting container because nvdiffrast requires CUDA
        runtime.run_lifting_module(
            "evaluation.scannetpp.gt_masks",
            [
                "--scene_root", str(self.scene_root),
                "--repo_root", str(runtime.repo_root),
                "--metadata", str(self.metadata_path),
                "--output_dir", str(output_dir),
                "--bands", str(bands),
            ],
        )
