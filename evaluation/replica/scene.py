# Replica scene loading, dataset conversion, visibility and GT masks

import json
import os
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from plyfile import PlyData

from ..common import SceneData, TargetClassInfo


CLASSES = [
    TargetClassInfo("chair", "chair", 57),
    TargetClassInfo("sofa", "couch", 58),
    TargetClassInfo("table", "dining table", 61),
    TargetClassInfo("tv", "tv", 63),
    TargetClassInfo("plant", "potted plant", 59),
    TargetClassInfo("clock", "clock", 75),
]

# Map main names to Replica names
REPLICA_CLASS_NAMES = {
    "chair": "chair",
    "sofa": "sofa",
    "table": "table",
    "tv": "tv-screen",
    "plant": "indoor-plant",
    "clock": "clock",
}

# Protocol values frozen in the manuscript: the pre-rendered sequence, one frame out of five,
# the fraction of incident faces a vertex label needs, and the depth tolerance of visibility
SEQUENCE_NAME = "Sequence_2"
FRAME_STEP = 5
VERTEX_LABEL_MIN_FRACTION = 0.6
VISIBILITY_SLOP = 0.05

# Intrinsics of Replica's pinhole camera
HEIGHT, WIDTH = 480, 640
FOCAL = 320.0
CX, CY = 320.0, 240.0

# Sparse point cloud sampling used when writing the COLMAP initial model
MAX_POINTS = 250000
FRAME_STRIDE = 10
PIXEL_STRIDE = 4
MAX_DEPTH_M = 10.0
SEED = 3


def _rotmat_to_qvec(rotation):
    """
    Converts a rotation matrix to COLMAP's quaternion convention using a symmetric matrix,
    its eigenvalues and eigenvectors, and the eigenvector of the largest eigenvalue.
    It flips the sign if the first component is negative to keep a stable convention.
    """
    rxx, ryx, rzx, rxy, ryy, rzy, rxz, ryz, rzz = rotation.flat
    matrix = np.array([
        [rxx - ryy - rzz, 0, 0, 0],
        [ryx + rxy, ryy - rxx - rzz, 0, 0],
        [rzx + rxz, rzy + ryz, rzz - rxx - ryy, 0],
        [ryz - rzy, rzx - rxz, rxy - ryx, rxx + ryy + rzz],
    ]) / 3.0
    values, vectors = np.linalg.eigh(matrix)
    qvec = vectors[[3, 0, 1, 2], np.argmax(values)]
    if qvec[0] < 0:
        qvec *= -1
    return qvec


def _world_to_camera(pose):
    """Invert a pose from camera coordinates to world coordinates"""
    rotation = pose[:3, :3]
    output = np.eye(4)
    output[:3, :3] = rotation.T

    # Rotate and negate the translation for camera coordinates
    output[:3, 3] = -rotation.T @ pose[:3, 3]
    return output


def _vertex_majority(n_vertices, faces, face_labels):
    """ Assign each vertex its most common face label when it reaches the threshold """
    # Count the number of votes for each label at each vertex, based on the labels of the faces that include that vertex
    votes = defaultdict(Counter)
    for face, label in zip(faces, face_labels):  # A face contains the indices of its vertices
        for vertex_index in set(face.tolist()):
            votes[vertex_index][int(label)] += 1

    # Vertices below the majority threshold remain invalid and are not annotated
    labels = np.full(n_vertices, -1, dtype=np.int64)
    for vertex_index, counts in votes.items():
        label, count = counts.most_common(1)[0]
        if count / sum(counts.values()) >= VERTEX_LABEL_MIN_FRACTION and label >= 0:
            labels[vertex_index] = label
    return labels


class ReplicaScene:
    """ Load Replica data and convert it to the common evaluation format """

    def __init__(self, data_root, scene, output_root):
        """
        Store the scene paths used by Replica processing

        - data_root: the root directory of the Replica dataset
        - scene: the name of the scene to process
        - output_root: the run directory, which holds the prepared dataset and the GT masks
        """
        self.data_root = Path(data_root)
        self.scene = scene
        self.scene_root = self.data_root / scene
        self.sequence = self.scene_root / scene / SEQUENCE_NAME
        self.prepared_dir = Path(output_root) / "dataset"
        self.masks_dir = Path(output_root) / "masks_gt2d"

    def selected_frames(self):
        """ Return the frame indices selected using the configured step """
        count = sum(1 for _ in open(self.sequence / "traj_w_c.txt"))
        return list(range(0, count, FRAME_STEP))

    def _load_mesh(self):
        """ Load vertices, faces and Replica dataset object IDs """
        # Replica stores semantic dataset IDs on mesh square faces
        ply = PlyData.read(str(self.scene_root / "mesh_semantic.ply"))
        vertex = ply["vertex"]

        # Load vertex coordinates
        vertices = np.vstack([vertex["x"], vertex["y"], vertex["z"]]).T.astype(np.float64)

        # Load face vertex indices (numpy returned an error if list was not used)
        faces = np.asarray([list(item) for item in ply["face"].data["vertex_indices"]], dtype=np.int64)

        # Load their corresponding Replica dataset object IDs, as every face has an object_id attribute, not the vertices
        face_instances_ids = np.asarray(ply["face"].data["object_id"], dtype=np.int64)
        return vertices, faces, face_instances_ids

    def _load_info(self):
        """
        Load the semantic class metadata for the current scene

        info_semantic.json contains the Replica class list ("classes") and the
        "id_to_label" table that converts instance IDs to semantic IDs.
        """
        return json.loads((self.scene_root / "info_semantic.json").read_text())

    def _dataset_ids_to_local_ids(self, info):
        """ Map Replica dataset semantic IDs to SceneData local IDs """
        # Map Replica names to Replica IDs
        name_to_id = {item["name"]: int(item["id"]) for item in info["classes"]}

        # Map Replica dataset IDs to SceneData local main IDs using REPLICA_CLASS_NAMES
        return {name_to_id[REPLICA_CLASS_NAMES[item.name]]: local_id
                for local_id, item in enumerate(CLASSES)
                if REPLICA_CLASS_NAMES[item.name] in name_to_id}

    def load_data(self):
        """Load mesh labels and visibility as common scene data."""
        # Load Replica geometry and convert every Replica dataset ID to a SceneData local ID in the main project vocabulary
        vertices, faces, face_instances_ids = self._load_mesh()
        info = self._load_info()
        dataset_ids_to_local_ids = self._dataset_ids_to_local_ids(info)

        # Converts the Replica instance id to the Replica semantic class id
        id_to_label = np.asarray(info["id_to_label"], dtype=np.int64)

        # Map Replica face instance IDs to Replica face dataset semantic IDs
        face_dataset = np.where((face_instances_ids >= 0) & (face_instances_ids < len(id_to_label)),
                            id_to_label[np.clip(face_instances_ids, 0, len(id_to_label) - 1)],
                            -1)

        # Convert Replica face dataset IDs to main local IDs for voting
        face_labels = np.asarray([dataset_ids_to_local_ids.get(int(value), -1)
                                  for value in face_dataset], dtype=np.int64)

        # Uses face_dataset, Replica dataset IDs to identify vertices with a source annotation, independently of the main local labels
        vertex_dataset = _vertex_majority(len(vertices), faces, face_dataset)

        # Convert main local face labels to main local vertex labels for metrics
        semantic = _vertex_majority(len(vertices), faces, face_labels)

        # Visibility is derived from RGB and depth frames
        return SceneData(
            dataset="replica",
            scene=self.scene,
            vertices=vertices,
            semantic_labels=semantic,
            annotated=(vertex_dataset >= 0),
            visible=self._visibility(vertices),
            classes=CLASSES,
        )

    def _load_trajectory(self):
        """ Load the sequence camera poses in world coordinates """
        return np.loadtxt(self.sequence / "traj_w_c.txt", dtype=np.float64).reshape(-1, 4, 4)

    def _load_depth(self, index):
        """ Load one depth image and convert its values to meters """
        # Replica stores depth in millimeters, so we convert it to meters here
        return np.asarray(Image.open(self.sequence / "depth" / f"depth_{index}.png"),
                          dtype=np.float64) * 0.001

    def _load_semantic_image(self, index):
        """ Load one Replica semantic image """
        # The semantic image uses Replica dataset IDs before conversion to the local ID space
        return np.asarray(Image.open(
            self.sequence / "semantic_class" / f"semantic_class_{index}.png"),
            dtype=np.int64,
        )

    def _visibility(self, vertices):
        """
        Calculate visible vertices supported by 2D observations

        A vertex is considered visible if:
            - It is projected into the camera's view frustum
            - It is in front of the camera
            - Its depth matches the observed depth within a certain tolerance.
        """

        # Load the camera trajectory and initialize a visibility mask for all vertices
        trajectory = self._load_trajectory()
        visible = np.zeros(len(vertices), dtype=bool)

        for index in self.selected_frames():

            # Represent mesh vertices in camera coordinates for the current frame
            pose = _world_to_camera(trajectory[index])
            camera_points = (pose[:3, :3] @ vertices.T).T + pose[:3, 3]
            z = camera_points[:, 2]

            # Project camera points with the pinhole model
            # Ignore invalid depth divisions
            with np.errstate(divide="ignore", invalid="ignore"):
                u = FOCAL * camera_points[:, 0] / z + CX
                v = FOCAL * camera_points[:, 1] / z + CY

            # Rounding the pixel coordinates to integer
            ui = np.round(u).astype(np.int64)
            vi = np.round(v).astype(np.int64)

            # Determine which vertices are projected inside the image boundaries and in front of the camera
            inside = ((z > 0) & (ui >= 0) & (ui < WIDTH) & (vi >= 0) & (vi < HEIGHT))
            candidates = np.where(inside)[0]
            if len(candidates) == 0:
                continue

            # Compare projected depth with the observed depth to reject occluded vertices,
            # allowing for a small tolerance defined by VISIBILITY_SLOP
            depth = self._load_depth(index)
            image_depth = depth[vi[candidates], ui[candidates]]
            hit = ((image_depth > 0) & (np.abs(z[candidates] - image_depth) <= VISIBILITY_SLOP))
            visible[candidates[hit]] = True

        return visible

    def prepare_dataset(self, runtime):
        """
        Prepare Replica images and COLMAP model for training

        COLMAP is a standard that consists of:
            - intrinsics in sparse/0/cameras.txt
            - extrinsics in sparse/0/images.txt
            - a sparse point cloud in sparse/0/points3D.txt

        The runtime is not needed, as everything is written on the host
        """

        # Training expects an images directory and a COLMAP model
        images_dir = self.prepared_dir / "images"
        sparse_dir = self.prepared_dir / "sparse" / "0"
        images_dir.mkdir(parents=True, exist_ok=True)
        sparse_dir.mkdir(parents=True, exist_ok=True)

        # Link or copy the selected RGB frames into the training directory
        trajectory = self._load_trajectory()
        frames = self.selected_frames()
        for index in frames:
            source = self.sequence / "rgb" / f"rgb_{index}.png"
            target = images_dir / f"rgb_{index}.png"
            if not target.exists():
                try:
                    os.symlink(os.path.relpath(source, target.parent), target)
                except OSError:
                    target.write_bytes(source.read_bytes())

        # Write the fixed Replica camera intrinsics
        (sparse_dir / "cameras.txt").write_text(f"# Camera list\n1 PINHOLE {WIDTH} {HEIGHT} {FOCAL} {FOCAL} {CX} {CY}\n")

        # Write selected camera poses in COLMAP format
        image_lines = ["# Image list\n"]
        for image_id, index in enumerate(frames, start=1):
            pose = _world_to_camera(trajectory[index])
            qvec = _rotmat_to_qvec(pose[:3, :3])
            translation = pose[:3, 3]

            image_lines.append(
                f"{image_id} {qvec[0]:.12f} {qvec[1]:.12f} {qvec[2]:.12f} "
                f"{qvec[3]:.12f} {translation[0]:.12f} {translation[1]:.12f} "
                f"{translation[2]:.12f} 1 rgb_{index}.png\n\n"
            )
        (sparse_dir / "images.txt").write_text("".join(image_lines))

        # COLMAP needs an initial sparse point cloud to start the reconstruction
        # Sample a point cloud from RGB and depth images for COLMAP
        rng = np.random.default_rng(SEED)
        points, colors = [], []
        for index in frames[::FRAME_STRIDE]:

            # For every sampled frame, load depth and rgb
            depth = self._load_depth(index)
            rgb = np.asarray(Image.open(self.sequence / "rgb" / f"rgb_{index}.png"))

            # We sample one out of pixel_stride pixels in each axis
            ys, xs = np.meshgrid(np.arange(0, HEIGHT, PIXEL_STRIDE), np.arange(0, WIDTH, PIXEL_STRIDE), indexing="ij")
            z = depth[ys, xs].reshape(-1)
            valid = (z > 0.01) & (z < MAX_DEPTH_M)

            # Inverse pinhole projection: 3D coordinates from pixel coordinates and depth
            x = (xs.reshape(-1) - CX) * z / FOCAL
            y = (ys.reshape(-1) - CY) * z / FOCAL
            camera_points = np.stack([x, y, z], axis=1)[valid]

            # Convert the camera coordinates to world coordinates using the camera pose
            world_points = (trajectory[index][:3, :3] @ camera_points.T).T + trajectory[index][:3, 3]

            # Save valid sampled points in world coordinates with their RGB colors
            colors.append(rgb[ys.reshape(-1)[valid], xs.reshape(-1)[valid]])
            points.append(world_points)

        points = np.concatenate(points)
        colors = np.concatenate(colors)
        if len(points) > MAX_POINTS:
            selected = rng.choice(len(points), MAX_POINTS, replace=False)
            points, colors = points[selected], colors[selected]

        # Save sampled points for COLMAP, the last file, whose presence marks the dataset as prepared
        with open(sparse_dir / "points3D.txt", "w") as output:
            output.write("# Point list\n")
            for point_id, (point, color) in enumerate(zip(points, colors), start=1):
                output.write(
                    f"{point_id} {point[0]:.6f} {point[1]:.6f} {point[2]:.6f} "
                    f"{int(color[0])} {int(color[1])} {int(color[2])} 1.0\n"
                )

    def generate_gt_masks(self, runtime):
        """
        Generate binary 2D GT masks from Replica semantic images

        The runtime is not needed, as the masks are written on the host
        """
        (self.masks_dir / "semantic").mkdir(parents=True, exist_ok=True)
        (self.masks_dir / "confidence").mkdir(parents=True, exist_ok=True)

        # Convert Replica semantic IDs into the stored detector IDs used by the mask pipeline
        dataset_semantic_to_detector_stored = {
            dataset_id: CLASSES[local_id].detector_stored_id
            for dataset_id, local_id in self._dataset_ids_to_local_ids(self._load_info()).items()
        }

        # Save one semantic and confidence pair per selected frame
        for frame in self.selected_frames():

            # Loads the Replica semantic 2D image for the current frame, which contains the dataset 2D GT semantic IDs for each pixel
            dataset = self._load_semantic_image(frame)
            mapped = np.zeros(dataset.shape, dtype=np.uint8)

            # Map Replica IDs to stored detector IDs
            for dataset_id, stored_id in dataset_semantic_to_detector_stored.items():
                mapped[dataset == dataset_id] = stored_id
            name = f"rgb_{frame}"
            cv2.imwrite(str(self.masks_dir / "semantic" / f"{name}.png"), mapped)
            cv2.imwrite(str(self.masks_dir / "confidence" / f"{name}.png"),
                        (mapped > 0).astype(np.uint8) * 255)

        # classes.json is written last, so its presence marks the masks as complete
        classes = {str(item.detector_stored_id): item.name_by_detector for item in CLASSES}
        (self.masks_dir / "classes.json").write_text(json.dumps(classes, indent=2))
