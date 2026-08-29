# CLI entry point for Scannet++ reference masks and visible vertex support in the lifting container

import argparse
import importlib.util
import json
from pathlib import Path

import cv2
import numpy as np
import torch
import nvdiffrast.torch as dr

from ..common import atomic_write_text, ensure_dir
from .scene import CLASSES, DATASET_LABELS, MASKS_CACHE_VERSION
from plyfile import PlyData

'''
Convert OpenCV camera coordinates to the OpenGL convention used by nvdiffrast

OpenCV uses x right, y down and z forward, while the OpenGL projection used by nvdiffrast uses x right, y up and
the opposite z direction before the perspective divide
'''
CV_TO_GL = np.diag([1.0, -1.0, -1.0, 1.0])


def _load_colmap_loader(repo_root):
    """
    Load the repository COLMAP reader
    """

    # Import only the reader module because the full training package has extra dependencies
    path = repo_root / "scene" / "colmap_loader.py"
    spec = importlib.util.spec_from_file_location("unified_colmap_loader", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _projection(fx, fy, cx, cy, width, height, near, far):
    """
    Build the OpenGL projection matrix for one image band

    The matrix converts pinhole camera coordinates into clip coordinates. The principal point is expressed
    relative to the current band, not the full image, because each band has its own vertical coordinate origin
    """

    # Build the matrix directly from the pinhole intrinsics and clipping planes
    matrix = np.zeros((4, 4), dtype=np.float64)

    # Convert the horizontal focal length from pixel units into normalized device coordinates
    matrix[0, 0] = 2.0 * fx / width

    # Convert the horizontal principal point to OpenGL coordinates
    matrix[0, 2] = 1.0 - 2.0 * cx / width

    # Convert the vertical focal length and band relative principal point to OpenGL coordinates
    matrix[1, 1] = 2.0 * fy / height
    matrix[1, 2] = 2.0 * cy / height - 1.0

    # Map camera space depth between the near and far clipping planes into clip space depth
    matrix[2, 2] = -(far + near) / (far - near)
    matrix[2, 3] = -2.0 * far * near / (far - near)

    # Set w so the rasterizer performs the perspective divide using camera space depth
    matrix[3, 2] = -1.0
    return matrix


def _load_cameras(repo_root, sparse_dir):
    """
    Load pinhole camera records from the prepared COLMAP model

    The returned records contain image names, image sizes, intrinsic matrices and transforms from world coordinates to camera coordinates
    """

    # Read binary COLMAP files when available and otherwise use their text versions
    # Both files describe the same sparse model: cameras provide intrinsics
    # while images provide the pose and filename for every rendered view
    loader = _load_colmap_loader(repo_root)
    if (sparse_dir / "cameras.bin").exists():
        cameras = loader.read_intrinsics_binary(str(sparse_dir / "cameras.bin"))
        images = loader.read_extrinsics_binary(str(sparse_dir / "images.bin"))
    else:
        cameras = loader.read_intrinsics_text(str(sparse_dir / "cameras.txt"))
        images = loader.read_extrinsics_text(str(sparse_dir / "images.txt"))

    # Normalize both COLMAP formats into the camera records used by nvdiffrast,
    # sorting image names for deterministic output without changing camera poses
    result = []
    for image in sorted(images.values(), key=lambda item: item.name):
        camera = cameras[image.camera_id]

        # Support the two pinhole models produced by the prepared Scannet++ datasets
        # The original DSLR model is fisheye, but COLMAP undistorter writes a pinhole model in undistorted_colmap, which is the model rendered
        if camera.model == "PINHOLE":
            fx, fy, cx, cy = camera.params
        elif camera.model == "SIMPLE_PINHOLE":
            fx = fy = camera.params[0]
            cx, cy = camera.params[1], camera.params[2]
        else:
            raise ValueError("the prepared COLMAP model must use a pinhole camera")

        # COLMAP stores a quaternion and translation from world coordinates to camera coordinates
        transform = np.eye(4)
        transform[:3, :3] = loader.qvec2rotmat(image.qvec)
        transform[:3, 3] = np.asarray(image.tvec)
        result.append({
            "name": image.name,
            "width": int(camera.width),
            "height": int(camera.height),

            # K maps camera coordinates (x, y, z) to pixels using the pinhole model
            "K": np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]]),
            "world_to_camera": transform,
        })
    return result


def _load_mesh(scene_root, metadata_path):
    """
    Load geometry and convert official Scannet++ object annotations to triangle mask IDs

    The official 2D pipeline rasterizes the unlabelled mesh, maps visible faces to
    vertex object IDs from segments.json and segments_anno.json, and then maps
    object labels to the detector IDs expected by this evaluator

    We are literally following the official Scannet++ 2D pipeline, so the output is compatible with the released 2D annotations
    That is why the semantic mesh is not used here, even though it contains semantic information.
    The 2D object annotation format is defined over the official geometric mesh and its segment indices
    """

    # Load the same geometric mesh used by the official Scannet++ 2D rasterization pipeline
    scans_dir = scene_root / "scans"
    ply = PlyData.read(str(scans_dir / "mesh_aligned_0.05.ply"))
    vertex = ply["vertex"]

    # Load aligned vertex coordinates in world coordinates in the order indexed by
    # segments json maps vertices to segment IDs
    vertices = np.stack([vertex["x"], vertex["y"], vertex["z"]], axis=1).astype(np.float32)

    # Load triangle topology because rasterization returns the face covering each pixel
    # and the official 2D helper transfers vertex properties through face indices
    faces = np.asarray(list(ply["face"].data["vertex_indices"]), dtype=np.int32)

    # Map each mesh vertex to a segment and verify the shared vertex domain
    segments = json.loads((scans_dir / "segments.json").read_text())
    segment_indices = np.asarray(segments["segIndices"], dtype=np.int64)
    if len(segment_indices) != len(vertices):
        raise ValueError("segments.json and the rasterized mesh use different vertex counts")

    # Read object labels and assigned segment IDs
    annotations = json.loads((scans_dir / "segments_anno.json").read_text())

    # Validate that object labels come from the same released Scannet++ taxonomy
    # Reject annotations from another scene or taxonomy release
    metadata_names = {
        line.strip().lower()
        for line in metadata_path.read_text().splitlines()
        if line.strip()
    }

    # Convert every supported Scannet++ spelling into the local target class index
    # Several raw labels can represent one evaluated class, for example office chair and armchair both map to the local class chair
    label_to_local_id = {
        name.lower(): main_id
        for main_id, item in enumerate(CLASSES)
        for name in DATASET_LABELS[item.name]
    }
    object_to_local_id = {}

    # Build the mapping from segment IDs to annotated object IDs before
    # project those object IDs onto vertices containing each segment
    segment_to_object_id = {}
    for group in annotations["segGroups"]:
        raw_object_id = group.get("objectId", group.get("id"))
        if raw_object_id is None:
            raise ValueError("an annotation group has no object ID")
        object_id = int(raw_object_id)
        if object_id < 0:
            raise ValueError(f"object {object_id} is invalid")

        label = str(group.get("label", "")).strip().lower()
        if label and label not in metadata_names:
            raise ValueError(f"object {object_id} uses unknown Scannet++ label: {label}")

        # Keep the object mapping separate so one label can cover all assigned segments
        object_to_local_id[object_id] = label_to_local_id.get(label, -1)
        segment_ids = np.asarray(group.get("segments", []), dtype=np.int64)
        for segment_id in segment_ids:
            segment_to_object_id[int(segment_id)] = object_id

    if np.any((faces < 0) | (faces >= len(vertices))):
        raise ValueError("the mesh contains a face with an invalid vertex index")

    # Convert segment IDs into one object ID per mesh vertex
    vertex_object_ids = np.asarray(
        [segment_to_object_id.get(int(segment_id), -1)
         for segment_id in segment_indices],
        dtype=np.int64,
    )

    # Convert annotated object IDs into local class IDs
    # This is the same semantic representation used by get_sem_ids_on_2d() in the official code before this project applies detector IDs
    vertex_local_ids = np.asarray([
        object_to_local_id.get(int(object_id), -1)
        for object_id in vertex_object_ids
    ], dtype=np.int64)

    '''
    Important decision:
    The official helper get_vtx_prop_on_2d() assigns a face property using the first vertex of each face.
    Keeping this rule makes the output comparable with the official Scannet++ 2D annotations

    Replica uses the majority vertex label of each face because it has no official 2D annotations
    '''

    tri_main = vertex_local_ids[faces[:, 0]]

    # Rasterization uses one stored detector ID per triangle in the global vocabulary
    # Zero remains background and ignored objects are not confused with any target class
    tri_stored = np.zeros(len(faces), dtype=np.uint8)
    for index, item in enumerate(CLASSES):
        tri_stored[tri_main == index] = item.detector_stored_id
    return vertices, faces, tri_stored


def generate(scene_root, repo_root, metadata_path, output_dir, bands=4,
             force=False, mask_version=MASKS_CACHE_VERSION, resolution=None):
    """
    Render reference masks and save visible vertex support data

    - bands: number of horizontal bands used to limit GPU memory
    - force: ignore existing masks and support data when enabled

    Each output semantic PNG contains detector stored IDs, not local class IDs
    The global evaluator consumes this format for detector predictions and dataset reference masks
    """
    if bands < 1:
        raise ValueError("bands must be at least 1")

    # Reuse the completed output unless the caller requests a rebuild
    cache_info_path = output_dir / "render_metadata.json"
    cache_info = None
    if cache_info_path.exists():
        cache_info = json.loads(cache_info_path.read_text())

    if ((output_dir / "classes.json").exists() and (output_dir / "support.npz").exists() and
            (output_dir / "camera_intrinsics.json").exists() and cache_info is not None and
            cache_info.get("version") == mask_version and
            cache_info.get("bands") == bands and
            cache_info.get("resolution") == resolution and not force):
        return output_dir

    # nvdiffrast renders these masks on CUDA rather than through the host CPU
    if not torch.cuda.is_available():
        raise RuntimeError("Scannet++ GT rendering requires a CUDA device")

    # Use images and a sparse model from the same undistortion run
    sparse_dir = scene_root / "dslr" / "undistorted_colmap" / "sparse" / "0"
    if not sparse_dir.exists():
        raise FileNotFoundError(f"prepared COLMAP model not found: {sparse_dir}")

    vertices, faces, tri_stored = _load_mesh(scene_root, metadata_path)
    cameras = _load_cameras(repo_root, sparse_dir)

    ensure_dir(output_dir / "semantic")
    ensure_dir(output_dir / "confidence")
    device = torch.device("cuda")

    # Homogeneous vertices allow one matrix multiplication per camera and band
    # Appending one makes the affine translation part of the matrix product
    vertices_h = torch.from_numpy(np.concatenate([vertices, np.ones((len(vertices), 1), dtype=np.float32)], axis=1)).to(device)
    # Upload faces once and reuse them for every camera and band
    faces_t = torch.from_numpy(faces).to(device).contiguous()

    # Labels are indexed by the triangle IDs returned by nvdiffrast
    labels_t = torch.from_numpy(tri_stored.astype(np.int64)).to(device)
    context = dr.RasterizeCudaContext()
    visible_vertices = np.zeros(len(vertices), dtype=bool)

    for camera in cameras:

        # Render each image in horizontal bands to limit GPU memory usage
        width, height = camera["width"], camera["height"]

        # COLMAP uses OpenCV camera coordinates, while nvdiffrast expects OpenGL camera coordinates
        transform = CV_TO_GL @ camera["world_to_camera"]

        # Derive clipping planes from camera space mesh depths instead of a scene wide heuristic
        # COLMAP's z coordinate is positive in front of the camera
        # Discard vertices behind the camera during projection
        camera_points = (camera["world_to_camera"][:3, :3] @ vertices.T).T + camera["world_to_camera"][:3, 3]
        positive_depth = camera_points[:, 2][camera_points[:, 2] > 1e-4]
        if len(positive_depth) == 0:
            raise ValueError(f"camera {camera['name']} cannot see any mesh vertex")
        near = max(1e-3, float(positive_depth.min()))
        far = max(near + 1.0, float(positive_depth.max()))

        # Split the image height into equally sized bands so intermediate CUDA buffers stay bounded
        edges = np.linspace(0, height, bands + 1).astype(int)
        rendered_bands = []
        for band in range(bands):

            # Shift the principal point for the current vertical image band because the band has its own image origin
            y0, y1 = int(edges[band]), int(edges[band + 1])
            band_height = y1 - y0
            projection = _projection(
                camera["K"][0, 0], camera["K"][1, 1], camera["K"][0, 2],
                camera["K"][1, 2] - y0, width, band_height, near, far,
            )

            # nvdiffrast expects the combined projection and camera transform transposed for row vector multiplication
            matrix = torch.from_numpy((projection @ transform).T.astype(np.float32)).to(device)
            clip_vertices = (vertices_h @ matrix).unsqueeze(0).contiguous()

            # Rasterize the mesh and recover nvdiffrast triangle indices, not semantic class IDs
            # The depth test keeps the frontmost triangle at each pixel
            raster, _ = dr.rasterize(
                context, clip_vertices, faces_t, resolution=[band_height, width],
            )

            # The fourth raster channel stores the one based triangle index
            # Convert the zero background index to an invalid index
            face_ids = raster[0, :, :, 3].round().long() - 1
            hit = face_ids >= 0

            # Any triangle contributing to a pixel makes all of its vertices visible in at least one camera view
            # This matches the support approximation
            visible_face_ids = torch.unique(face_ids[hit]).cpu().numpy()
            if len(visible_face_ids):
                visible_faces = faces[visible_face_ids].reshape(-1)
                visible_vertices[visible_faces] = True

            # Convert visible triangle indices into stored detector class IDs
            # Keep unknown objects as background
            band_labels = torch.zeros((band_height, width), dtype=torch.int64, device=device)
            band_labels[hit] = labels_t[face_ids[hit]]

            # Restore the image row order after rendering the band in OpenGL coordinates
            rendered_bands.append(torch.flip(band_labels, dims=(0,)).to(torch.uint8))

        # Join the vertically rendered bands back into one image mask size
        semantic = torch.cat(rendered_bands, dim=0).cpu().numpy()

        # Confidence follows the existing mask contract: every nonbackground pixel is a confident reference pixel
        confidence = (semantic > 0).astype(np.uint8) * 255
        stem = Path(camera["name"]).stem
        cv2.imwrite(str(output_dir / "semantic" / f"{stem}.png"), semantic)
        cv2.imwrite(str(output_dir / "confidence" / f"{stem}.png"), confidence)

    # Save detector names keyed by stored IDs for the evaluator and vote stage
    classes = {str(item.detector_stored_id): item.name_by_detector for item in CLASSES}
    (output_dir / "classes.json").write_text(json.dumps(classes, indent=2))

    # Save the rendered COLMAP intrinsics
    (output_dir / "camera_intrinsics.json").write_text(json.dumps([
        {
            "name": camera["name"],
            "width": camera["width"],
            "height": camera["height"],
            "fx": float(camera["K"][0, 0]),
            "fy": float(camera["K"][1, 1]),
            "cx": float(camera["K"][0, 2]),
            "cy": float(camera["K"][1, 2]),
        }
        for camera in cameras
    ], indent=2))

    # Save visibility in scene vertex order
    np.savez_compressed(
        output_dir / "support.npz",
        visible_vertices=visible_vertices,
    )

    # Record the conversion contract so later runs can reject stale outputs
    atomic_write_text(cache_info_path, json.dumps({
        "version": int(mask_version),
        "mesh": "mesh_aligned_0.05.ply",
        "annotations": ["segments.json", "segments_anno.json"],
        "bands": int(bands),
        "resolution": resolution,
        "source": str(metadata_path),
    }, indent=2) + "\n")
    return output_dir


def main():
    """
    Generate Scannet++ reference masks from command arguments

    The CLI is called by run.py inside the lifting container, which can use CUDA
    The metadata path is retained as an explicit input because it validates that
    segment annotation labels belong to the released Scannet++ taxonomy
    """
    parser = argparse.ArgumentParser()

    # Identify the scene, repository reader, metadata and output directory for rendering
    parser.add_argument("--scene_root", required=True, type=Path)
    parser.add_argument("--repo_root", required=True, type=Path)
    parser.add_argument("--metadata", required=True, type=Path)
    parser.add_argument("--output_dir", required=True, type=Path)
    parser.add_argument("--bands", type=int, default=4)
    parser.add_argument("--mask_version", type=int, default=MASKS_CACHE_VERSION)
    parser.add_argument("--resolution", type=int, default=None)

    # Render masks and support data again when completion files already exist
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    generate(args.scene_root, args.repo_root, args.metadata, args.output_dir,
             args.bands, args.force, args.mask_version, args.resolution)


if __name__ == "__main__":
    main()
