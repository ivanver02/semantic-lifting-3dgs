# Transfer labels between mesh vertices and Gaussians using radius votes

import itertools

import numpy as np
from scipy.spatial import cKDTree
from plyfile import PlyData

EPS = 1e-10
RADIUS_NEIGHBOR_CHUNK_SIZE = 100000


def load_gaussian_ply(path):
    """
    Load Gaussian centers and opacity values converted to alpha

    The return value contains one array of 3D centers and one array of opacity values in the same Gaussian order
    """

    vertex = PlyData.read(str(path))["vertex"]
    xyz = np.vstack([vertex["x"], vertex["y"], vertex["z"]]).T.astype(np.float64)

    # Gaussian PLY files store opacity before the sigmoid conversion
    opacity = 1.0 / (1.0 + np.exp(-np.asarray(vertex["opacity"], dtype=np.float64)))
    return xyz, opacity


def build_radius_neighbors(vertices, gaussians, radius):
    """
    Build every pair of a mesh vertex and a Gaussian center closer than radius

    The result contains three arrays of the same length: vertex indices, Gaussian indices and distances.
    The pairs are symmetric, so they serve both transfer directions
    """

    tree = cKDTree(gaussians)
    vertex_chunks, gaussian_chunks, distance_chunks = [], [], []

    # Query the Gaussian KD tree in chunks so large scenes do not require one huge query
    for start in range(0, len(vertices), RADIUS_NEIGHBOR_CHUNK_SIZE):
        lists = tree.query_ball_point(vertices[start:start + RADIUS_NEIGHBOR_CHUNK_SIZE], r=radius)

        # Flatten the variable length neighbor lists, repeating each vertex once per neighbor
        counts = np.fromiter((len(item) for item in lists), dtype=np.int64, count=len(lists))
        vertex_index = np.repeat(np.arange(start, start + len(lists), dtype=np.int32), counts)
        gaussian_index = np.fromiter(itertools.chain.from_iterable(lists), dtype=np.int32, count=int(counts.sum()))

        vertex_chunks.append(vertex_index)
        gaussian_chunks.append(gaussian_index)
        distance_chunks.append(np.linalg.norm(
            gaussians[gaussian_index] - vertices[vertex_index], axis=1).astype(np.float32))

    return (np.concatenate(vertex_chunks), np.concatenate(gaussian_chunks),
            np.concatenate(distance_chunks))


def radius_label_vote(n_query, query, reference, distances, reference_labels, reference_weights,
                      n_classes, min_fraction, background_labels_compete):
    """
    Assign local labels to query points with a weighted radius vote and optional abstention

    This function works in both directions, from mesh vertices to Gaussians and from Gaussians to mesh vertices
    When Gaussians to mesh, the reference labels are the local IDs of the Gaussians and the query points are mesh vertices

    More precisely, for each query point (mesh vertex), we have a set of neighboring reference points (Gaussians) with known local labels and weights
    We want to assign a local label to each query point based on the weighted votes of its neighbors

    query, reference and distances hold one entry per neighboring pair
    Reference labels are local IDs in [0, n_classes) or -1 for the background, and they use the order of the referenced points, as the weights
    background_labels_compete decides whether non-target labels, including -1, contribute to the vote denominator
    """

    # Expand edges into one row per neighboring reference point
    edge_labels = reference_labels[reference]
    edge_weights = reference_weights[reference] / (distances.astype(np.float64) ** 2 + EPS) # Closer neighbors have more influence
    if not background_labels_compete:
        edge_weights = np.where(edge_labels >= 0, edge_weights, 0.0)

    '''
    Imagine:
      Edge   Query   Reference   Label   Weight

         0       0           3       0      100
         1       0           7       0       25
         2       0          10       1     6.25
         3       1           2       1    11.11
         4       1           4       0      100

    For query 0, the total weight is 100 + 25 + 6.25 = 131.25
    Then, total[0] = 131.25
    And repeats for every query vertex
    '''

    # Compute the competing evidence denominator
    total = np.bincount(query, weights=edge_weights, minlength=n_query) # Weighted sum per binning query vertex

    '''
    For query 0: (one cell in the bincount)
        The score for label 0 is 100 + 25 = 125
        The score for label 1 is 6.25
        The fraction for label 0 is 125 / 131.25 = 0.952
        The fraction for label 1 is 6.25 / 131.25 = 0.048
        If min_fraction = 0.5, then label 0 is accepted and label 1 is rejected. The query vertex receives label 0
        If min_fraction = 0.99, then both labels are rejected and the query vertex receives -1
    '''

    # Sum the weights per query and label, the last column (label -1) holds the background
    columns = n_classes + 1
    scores = np.bincount(query.astype(np.int64) * columns + edge_labels % columns,
                         weights=edge_weights, minlength=n_query * columns).reshape(n_query, columns)

    # Select the strongest class score for each query
    best_label = scores[:, :n_classes].argmax(axis=1)
    best_score = scores[np.arange(n_query), best_label]

    # The class is accepted when its fraction of the competing evidence reaches min_fraction, which
    # is the only rule, so a value below 0.5 can label a vertex where the background holds more weight
    fraction = np.divide(best_score, total, out=np.zeros_like(best_score), where=total > 0)
    accepted = (best_score > 0) & (fraction >= min_fraction)
    return np.where(accepted, best_label, -1)


# The nearest pair of every query only depends on the pairs, not on the labels, so it is kept for the
# pairs it was computed from. An evaluation transfers every class and beta through the same pairs
_nearest_cache = {"pairs": None, "nearest": None}


def _nearest_pairs(query, reference, distances):
    """ Index of the nearest pair of every query that has one, computed once for the same pair arrays """
    pairs = (query, reference, distances)
    cached = _nearest_cache["pairs"]
    if cached is not None and all(a is b for a, b in zip(cached, pairs)):
        return _nearest_cache["nearest"]

    # Sort the pairs by query and then by distance, so the first pair of each query is its nearest reference
    order = np.lexsort((distances, query))
    _, first = np.unique(query[order], return_index=True)
    nearest = order[first]
    _nearest_cache.update(pairs=pairs, nearest=nearest)
    return nearest


def nearest_neighbor_label(n_query, query, reference, distances, reference_labels):
    """ Assign each query point the label of its nearest reference point within the radius """
    nearest = _nearest_pairs(query, reference, distances)
    output = np.full(n_query, -1, dtype=np.int64)
    output[query[nearest]] = reference_labels[reference[nearest]]
    return output


def predict_vertex_labels(neighbors, n_vertices, selected, gaussian_opacity, min_fraction,
                          opacity_weighted, min_opacity, gaussian_to_mesh_background_competes,
                          gaussian_to_mesh_transfer):
    """
    Transfer a binary Gaussian selection to mesh vertices with the selected method

    opacity_weighted decides whether Gaussian opacity affects the vote weights
    gaussian_to_mesh_background_competes decides whether background neighbors compete with target labels
    gaussian_to_mesh_transfer selects radius voting or assignment to the nearest point
    Returns a boolean mask of the vertices that receive the class
    """

    vertex_index, gaussian_index, distances = neighbors

    # The selected Gaussians carry the class (label 0), the rest are background
    labels = np.where(selected, 0, -1)
    if gaussian_to_mesh_transfer == "nearest_neighbor_label":
        return nearest_neighbor_label(n_vertices, vertex_index, gaussian_index, distances, labels) == 0

    # Select opacity weight or uniform weights before running the vote
    if opacity_weighted:
        weights = np.maximum(gaussian_opacity, min_opacity)
    else:
        weights = np.ones(len(labels), dtype=np.float64)

    return radius_label_vote(
        n_vertices, vertex_index, gaussian_index, distances, labels, weights, 1,
        min_fraction, gaussian_to_mesh_background_competes,
    ) == 0


def ground_truth_gaussian_labels(neighbors, n_gaussians, scene, min_fraction,
                                 mesh_to_gaussian_background_competes):
    """
    Transfer the mesh annotation to the Gaussians through the same radius vote that carries the prediction back to the mesh

    mesh_to_gaussian_background_competes controls whether non-target mesh labels
    participate when transferring GT labels from the mesh to Gaussians.
    """

    vertex_index, gaussian_index, distances = neighbors

    # A vertex has no opacity, so every source votes with weight one
    return radius_label_vote(
        n_gaussians, gaussian_index, vertex_index, distances, scene.semantic_labels,
        np.ones(len(scene.vertices), dtype=np.float64), len(scene.classes),
        min_fraction, mesh_to_gaussian_background_competes,
    )
