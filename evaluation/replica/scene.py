# Replica scene loading, dataset conversion, visibility and GT masks

import json
import os
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from plyfile import PlyData

from ..common import SceneData, TargetClassInfo, atomic_write_text, ensure_dir



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

# Convert a rotation matrix to a quaternion
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

    # Complete the quaternion matrix
        [rzx + rxz, rzy + ryz, rzz - rxx - ryy, 0],
        [ryz - rzy, rzx - rxz, rxy - ryx, rxx + ryy + rzz],
    ]) / 3.0
    values, vectors = np.linalg.eigh(matrix)
    qvec = vectors[[3, 0, 1, 2], np.argmax(values)]
    if qvec[0] < 0:
        qvec *= -1
    return qvec

    # Return the quaternion


class ReplicaScene:
    """ Load Replica data and convert it to the common evaluation format """

    def __init__(self, data_root, scene, sequence_name, frame_step, seed,
                 vertex_label_min_fraction, visibility_slop):
        """ 
        Store the scene paths and thresholds used by Replica processing
        
        - data_root: the root directory of the Replica dataset
        - scene: the name of the scene to process
        - sequence_name: the name of the sequence to process
        - frame_step: the step size for selecting frames from the sequence
        - seed: the random seed for sampling points
        - vertex_label_min_fraction: the minimum fraction of face labels required for a vertex to be annotated
        - visibility_slop: the maximum allowed depth difference for a visible vertex
        """

    # Store the scene configuration
        self.data_root = Path(data_root)
        self.scene = scene
        self.scene_root = self.data_root / scene
        self.sequence = self.scene_root / scene / sequence_name
        self.frame_step = frame_step
        self.seed = seed
        self.vertex_label_min_fraction = vertex_label_min_fraction
        self.visibility_slop = visibility_slop

    # Complete the metadata example

    def selected_frames(self):
        """ Return the frame indices selected using the configured step """
        count = sum(1 for _ in open(self.sequence / "traj_w_c.txt"))
        return list(range(0, count, self.frame_step))

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

        It is something like this:

        {
            "classes": [{
                        "children": [],
                        "id": 1,
                        "name": "backpack",
                        "parents": []
                        },

                        {
                        "children": [],
                        "id": 2,
                        "name": "base-cabinet",
                        "parents": []
                        }],
            omitted fields
        }

    # Add visibility and class fields

        """
        # This file can give a map from Replica dataset semantic IDs to Replica dataset names
        return json.loads((self.scene_root / "info_semantic.json").read_text())

    def _dataset_ids_to_local_ids(self, info):
        """ Map Replica dataset semantic IDs to SceneData local IDs """
        # Map Replica names to Replica IDs
        name_to_id = {item["name"]: int(item["id"]) for item in info["classes"]}

        # Map main names to Replica dataset IDs
        # Uses the REPLICA_CLASS_NAMES dictionary that maps main class names to Replica dataset names
        dataset_names = {name: name_to_id.get(dataset_name, -1)
                     for name, dataset_name in REPLICA_CLASS_NAMES.items()}

        # Map Replica dataset IDs to SceneData local main IDs
        return {dataset_id: localID for localID, item in enumerate(CLASSES)
                for dataset_id in [dataset_names[item.name]] if dataset_id >= 0}

    @staticmethod
    def _vertex_majority(n_vertices, faces, face_labels, minimum):
        """ Assign each vertex its most common face label when it reaches the threshold """
        # Convert face labels to vertex labels
        votes = {}

        # Count the number of votes for each label at each vertex, based on the labels of the faces that include that vertex
        for face_index, face in enumerate(faces): # A face contains the indices of its vertices
            label = int(face_labels[face_index])
            for vertex_index in np.unique(face):
                values = votes.setdefault(int(vertex_index), {})
                values[label] = values.get(label, 0) + 1

        # Vertices below the majority threshold remain invalid and are not annotated
        labels = np.full(n_vertices, -1, dtype=np.int64)
        for vertex_index, values in votes.items():
            label, count = max(values.items(), key=lambda item: item[1])
            if count / sum(values.values()) >= minimum and label >= 0:
                labels[vertex_index] = label
        return labels

    @staticmethod
    def _world_to_camera(pose):
        """Invert a pose from camera coordinates to world coordinates"""
        rotation = pose[:3, :3]
        output = np.eye(4)
        output[:3, :3] = rotation.T

        # Rotate and negate the translation for camera coordinates
        output[:3, 3] = -rotation.T @ pose[:3, 3]
        return output

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
        vertex_dataset = self._vertex_majority(
            len(vertices), faces, face_dataset, self.vertex_label_min_fraction,
        )

        # Convert main local face labels to main local vertex labels for metrics
        # Uses face_labels, which are already converted to main local IDs, to identify vertices with a main local label
        semantic = self._vertex_majority(
            len(vertices), faces, face_labels, self.vertex_label_min_fraction,
        )

        # Visibility is derived from RGB and depth semantic frames
        visible = self._visibility(vertices)
        selected_frames = self.selected_frames()
        return SceneData(
            dataset="replica",
            scene=self.scene,
            vertices=vertices,
            semantic_labels=semantic,
            annotated=(vertex_dataset >= 0),

    # Complete the scene record
            visible=visible,
            classes=CLASSES,
            num_images=len(selected_frames),
            camera_intrinsics=[
                {"width": 640, "height": 480, "fx": 320.0, "fy": 320.0,
                 "cx": 320.0, "cy": 240.0}
                for _ in selected_frames
            ],
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
        Calculate visible vertices supported by 2D labels 
        
        A vertex is considered visible if:
            - It is projected into the camera's view frustum
            - It is in front of the camera
            - Its depth matches the observed depth within a certain tolerance. 
            
        """

        # Load the camera trajectory and initialize a visibility mask for all vertices
        trajectory = self._load_trajectory()
        visible = np.zeros(len(vertices), dtype=bool)
        frame_indices = self.selected_frames()

        # Intrinsics of Replica's pinhole camera
        height, width = 480, 640
        fx = fy = 320.0
        cx, cy = 320.0, 240.0

        for index in frame_indices:

            # Represent mesh vertices in camera coordinates for the current frame
            pose = self._world_to_camera(trajectory[index])
            camera_points = (pose[:3, :3] @ vertices.T).T + pose[:3, 3]
            z = camera_points[:, 2]

            # Project camera points with the pinhole model
            # Ignore invalid depth divisions
            with np.errstate(divide="ignore", invalid="ignore"):
                u = fx * camera_points[:, 0] / z + cx
                v = fy * camera_points[:, 1] / z + cy

            # Rounding the pixel coordinates to integer
            ui = np.round(u).astype(np.int64)
            vi = np.round(v).astype(np.int64)

            # Determine which vertices are projected inside the image boundaries and in front of the camera
            inside = ((z > 0) & (ui >= 0) & (ui < width) & (vi >= 0) & (vi < height))
            candidates = np.where(inside)[0]
            if len(candidates) == 0:
                continue

            # Compare projected depth with the observed depth to reject occluded vertices
            depth = self._load_depth(index)
            image_depth = depth[vi[candidates], ui[candidates]]

            # Compare the projected depth with the observed depth, allowing for a small tolerance defined by visibility_slop
            hit = ((image_depth > 0) & (np.abs(z[candidates] - image_depth) <= self.visibility_slop))
            selected = candidates[hit]
            visible[selected] = True

        return visible

    def prepare_dataset(self, output_dir, max_points=250000, frame_stride=10, pixel_stride=4, max_depth_m=10.0):
        """ 
        Prepare Replica images and COLMAP model for training 
        
        COLMAP is a standard that consists of:
            - intrinsics in sparse/0/cameras.txt
            - extrinsics in sparse/0/images.txt
            - a sparse point cloud in sparse/0/points3D.txt
        """

        # Training expects an images directory and a COLMAP model
        images_dir = output_dir / "images"
        sparse_dir = output_dir / "sparse" / "0"
        required = [images_dir, sparse_dir]

        # Reuse the prepared dataset when all three COLMAP text files exist
        if all((output_dir / item).exists() for item in
               ["sparse/0/cameras.txt", "sparse/0/images.txt", "sparse/0/points3D.txt"]):
            return output_dir
        for path in required:
            ensure_dir(path)

        # Link or copy the selected RGB frames into the training directory
        trajectory = self._load_trajectory()
        frames = self.selected_frames()
        for index in frames:
            source = self.sequence / "rgb" / f"rgb_{index}.png"
            target = images_dir / f"rgb_{index}.png"
            if not target.exists():
                try:
                    os.symlink(os.path.relpath(source, target.parent), target)

    # Finish the point cloud output
                except OSError:
                    target.write_bytes(source.read_bytes())

        # Write the fixed Replica camera intrinsics
        (sparse_dir / "cameras.txt").write_text("# Camera list\n1 PINHOLE 640 480 320.0 320.0 320.0 240.0\n")

        # Write selected camera poses in COLMAP format
        image_lines = ["# Image list\n"]
        for image_id, index in enumerate(frames, start=1):
            pose = self._world_to_camera(trajectory[index])
            qvec = _rotmat_to_qvec(pose[:3, :3])
            translation = pose[:3, 3]

            # Write the image line
            image_lines.append(
                f"{image_id} {qvec[0]:.12f} {qvec[1]:.12f} {qvec[2]:.12f} "
                f"{qvec[3]:.12f} {translation[0]:.12f} {translation[1]:.12f} "
                f"{translation[2]:.12f} 1 rgb_{index}.png\n\n"
            )
        (sparse_dir / "images.txt").write_text("".join(image_lines))

        # COLMAP needs an initial sparse point cloud to start the reconstruction
        # Sample a point cloud from RGB and depth images for COLMAP
        rng = np.random.default_rng(self.seed)
        points, colors = [], []

        # We iterate every frame_stride frames, sample does not need to be enormous
        for index in frames[::frame_stride]:

            # For every selected frame, load depth and rgb
            depth = self._load_depth(index)
            rgb = np.asarray(Image.open(self.sequence / "rgb" / f"rgb_{index}.png"))

            # We sample one out of pixel_stride pixels in each axis
            ys, xs = np.meshgrid(np.arange(0, 480, pixel_stride), np.arange(0, 640, pixel_stride), indexing="ij")
            z = depth[ys, xs].reshape(-1) # We sample their depth
            valid = (z > 0.01) & (z < max_depth_m) # Filter some invalid values

            # Inverse formula of the pinhole projection: now we get the 3D coordinates from the pixel coordinates and depth
            x = (xs.reshape(-1) - 320.0) * z / 320.0
            y = (ys.reshape(-1) - 240.0) * z / 320.0
            camera_points = np.stack([x, y, z], axis=1)[valid]

            # Convert the camera coordinates to world coordinates using the camera pose
            world_points = (trajectory[index][:3, :3] @ camera_points.T).T + trajectory[index][:3, 3]

            # Save valid sampled points in world coordinates with their RGB colors
            colors.append(rgb[ys.reshape(-1)[valid], xs.reshape(-1)[valid]])
            points.append(world_points)

        # Concatenate all sampled points and colors from the selected frames into single arrays
        points = np.concatenate(points) # Appends arrays maintaining its shape
        colors = np.concatenate(colors)
        if len(points) > max_points:
            selected = rng.choice(len(points), max_points, replace=False)
            points, colors = points[selected], colors[selected]

        # Save sampled points in world coordinates with their RGB colors for COLMAP
        with open(sparse_dir / "points3D.txt", "w") as output:
            output.write("# Point list\n")
            for point_id, (point, color) in enumerate(zip(points, colors), start=1):
                output.write(
                    f"{point_id} {point[0]:.6f} {point[1]:.6f} {point[2]:.6f} "
                    f"{int(color[0])} {int(color[1])} {int(color[2])} 1.0\n"
                )
        return output_dir

    # Resolve mask metadata paths

    def generate_gt_masks(self, output_dir, force=False, resolution=None):
        """
        Generate or reuse binary 2D GT masks from Replica semantic images

        force regenerates the masks when enabled
        """

        metadata = {
            "version": 1,
            "sequence_name": self.sequence.name,
            "frame_step": self.frame_step,
            "vertex_label_min_fraction": self.vertex_label_min_fraction,
            "visibility_slop": self.visibility_slop,
            "resolution": resolution,
        }

    # Prepare mask output directories
        metadata_path = output_dir / "mask_metadata.json"
        previous = None
        if metadata_path.exists():
            previous = json.loads(metadata_path.read_text())
        if (output_dir / "classes.json").exists() and previous == metadata and not force:
            return output_dir
        ensure_dir(output_dir / "semantic")
        ensure_dir(output_dir / "confidence")

    # Complete mask metadata

        # Convert Replica semantic IDs into the stored detector IDs used by the mask pipeline
        info = self._load_info()
        dataset_ids_to_local_ids = self._dataset_ids_to_local_ids(info)

        # Convert local IDs to stored detector IDs
        local_to_detector_stored = {index: item.detector_stored_id for index, item in enumerate(CLASSES)}
        dataset_semantic_to_detector_stored = {dataset_id: local_to_detector_stored[local_id] for dataset_id, local_id in dataset_ids_to_local_ids.items()}
        
        # Save one semantic and confidence pair per selected frame
        for frame in self.selected_frames():

            # Loads the Replica semantic 2D image for the current frame, which contains the dataset 2D GT semantic IDs for each pixel
            dataset = self._load_semantic_image(frame)
            mapped = np.zeros(dataset.shape, dtype=np.uint8)

            # Map Replica IDs to stored detector IDs
            for dataset_id, stored_id in dataset_semantic_to_detector_stored.items():
                mapped[dataset == dataset_id] = stored_id
            name = f"rgb_{frame}"
            cv2.imwrite(str(output_dir / "semantic" / f"{name}.png"), mapped)
            cv2.imwrite(str(output_dir / "confidence" / f"{name}.png"),
                        (mapped > 0).astype(np.uint8) * 255)
            
        classes = {str(item.detector_stored_id): item.name_by_detector for item in CLASSES}
        (output_dir / "classes.json").write_text(json.dumps(classes, indent=2))
        atomic_write_text(metadata_path, json.dumps(metadata, indent=2) + "\n")
        return output_dir