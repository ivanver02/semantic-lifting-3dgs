# Data structures for metrics evaluation

import hashlib
import json
import os
import tempfile
from pathlib import Path

class TargetClassInfo:
    """
    Description of one target class in the main project vocabulary and its detector representation

    name is the main name used by the scene and metrics. It is not a dataset name and it is not the detector's name
    name_by_detector is the detector name written in classes.json
    detector_stored_id is the detector mask ID stored in PNG masks. It is the detector model ID shifted by one so zero remains the background ID
    """

    def __init__(self, name, name_by_detector, detector_stored_id):
        """Store class names and the detector mask ID"""
        self.name = name
        self.name_by_detector = name_by_detector
        self.detector_stored_id = detector_stored_id


class SceneData: # Created when loading data in the scene files
    """
    Ground truth and visibility data in the common scene representation

    The arrays related to vertex all use the same order of vertices in the mesh
    - annotated says whether the source dataset provides a semantic annotation
    - visible says if the vertex was observed by any of the selected camera views
    - classes contains the TargetClassInfo classes evaluated by the pipeline
    """

    def __init__(self, dataset, scene, vertices, semantic_labels,
                 annotated, visible, classes, num_images=0,
                 camera_intrinsics=None):
        """ 
        Store the scene names, vertex arrays and target classes 
        
        - dataset: The dataset name
        - scene: The scene name within the dataset
        - vertices: 3D coordinates of the mesh vertices in the scene
        - semantic_labels: array representing vertices with the local class ID annotation for each vertex
        - annotated: Boolean mask for vertices with one semantic label (maybe not in the target classes) in the source dataset
        - visible: Boolean mask for vertices observed by the selected camera views
        - classes: List of TargetClassInfo objects describing the target classes evaluated by the pipeline
        """
        self.dataset = dataset
        self.scene = scene
        self.vertices = vertices
        self.semantic_labels = semantic_labels
        self.annotated = annotated
        self.visible = visible
        self.classes = classes

    # Store image and camera metadata
        self.num_images = int(num_images)
        self.camera_intrinsics = list(camera_intrinsics or [])

    @property
    def class_ids(self):
        """ Map each main class name to its local scene ID """
        return {item.name: local_id for local_id, item in enumerate(self.classes)}

    @property
    def evaluation_mask(self):
        """
        Returns a boolean mask for vertices that should be included in evaluation
        
        Consequences of defining the evaluation mask this way:
        - Vertices that are not annotated or not visible are excluded from evaluation.
        - Vertices that are annotated but not in the target classes are included in evaluation, and can cause false positives.
        """

        # Visibility and annotation are independent conditions for evaluation
        return self.annotated & self.visible

    def class_id(self, name):
        """ Return the local ID for a main class name """
        return self.class_ids[name]


def safe_name(name):
    """ Make a detector name safe for a path """
    return name.replace(" ", "_")


def float_token(value):
    """ Format a float for a path """
    return str(value).replace(".", "_")


def main_digest(values, length=12):
    """ Return a digest for a configuration mapping """
    payload = json.dumps(values, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:length]


def vote_scope(parameters):
    """ Return settings for a vote artifact """
    keys = (
        "background_mode", "background_confidence", "background_view_policy",
        "raster_block_size", "vote_data_device",
    )
    return {key: parameters[key] for key in keys}


def vote_id(parameters):
    """ Return the id of a vote configuration """
    return "v" + main_digest(vote_scope(parameters))


def vote_class_dir(segmentation_dir, spec, identifier):
    return Path(segmentation_dir) / safe_name(spec.name_by_detector) / identifier


def ensure_dir(path):
    """ Create a directory and return its path """
    path.mkdir(parents=True, exist_ok=True)
    return path


def atomic_write_text(path, text):
    """ Replace a text artifact atomically """
    path = Path(path)
    ensure_dir(path.parent)
    fd, name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp",
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:

    # Write and flush the temporary file
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def target_classes_by_detector(classes):
    """ Map detector names to class records """
    return {item.name_by_detector: item for item in classes}