import numpy as np
from tqdm import tqdm
from typing import Union, List, Optional, Tuple, Any, Sequence
import sklearn
from sklearn.linear_model import ElasticNet, ElasticNetCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from sklearn.ensemble import RandomForestRegressor
import seaborn as sns
from neuroposelib.embed import Watershed
import time
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra, minimum_spanning_tree
from scipy.spatial import distance
import numpy.typing as npt
import pandas as pd


def get_z_scores(
    freq: npt.NDArray[Any],
    meta: pd.DataFrame,
    groups: Union[str, List[str]],
) -> Tuple[npt.NDArray[np.float64], List[Any]]:
    """
    Compute pairwise z-scores of cluster frequency means between groups.

    Parameters
    ----------
    freq : ndarray, shape (n_samples, n_clusters)
        Frequency/occupancy values per sample for each cluster.
    meta : pandas.DataFrame, shape (n_samples, ...)
        Metadata table aligned with `freq`. Rows correspond to samples.
    groups : str or list of str
        Column name(s) in `meta` to group by (e.g., condition or subject).

    Returns
    -------
    z_scores : ndarray(float64), shape (n_groups, n_groups, n_clusters)
        Pairwise z-scores comparing mean cluster frequencies between groups.
        `z_scores[i, j, c]` is (mean_group_i - mean_group_j) / pooled_std.
    labels : list
        Ordered list of the group labels corresponding to z_scores axes.
    """
    cluster_list = list(np.arange(freq.shape[-1], dtype=int))
    freq_df = pd.DataFrame(freq)
    freq_df.loc[:, 0] = 0
    freq_df = pd.concat([meta.loc[:, groups], freq_df], axis=1)

    meanz = freq_df[cluster_list + groups].groupby(groups).mean()
    stdz = freq_df[cluster_list + groups].groupby(groups).std()
    n_s = freq_df.groupby(groups).size().values
    mean_diffs = np.array(meanz)[:, None, ...] - np.array(meanz)[None, ...]
    std_diffs = np.sqrt(
        np.array(stdz)[:, None, ...] ** 2 / n_s[:, None, None]
        + np.array(stdz)[None, ...] ** 2 / n_s[None, :, None]
    )
    z_scores = mean_diffs / np.where(std_diffs == 0, 1, std_diffs)
    labels = list(meanz.index)

    return np.asarray(z_scores).astype(np.float64), labels

# def get_nn_graph(X: npt.NDArray, k: int = 5, weighted: bool = True) -> csr_matrix:
#     """Get nearest neighbor graph.

#     Parameters
#     ----------
#     X : npt.NDArray
#         Data array (# samples, # dimensions).
#     k : int, optional
#         Number of nearest neighbors.
#     weighted : bool, optional
#         If true, returns graph with edges weighted by Euclidean distances. Otherwise, all edges are unit distance.

#     Returns
#     -------
#     graph : csr_matrix
#         Nearest neighbor graph.
#     """    
#     X = np.ascontiguousarray(X, dtype=np.float32)

#     # max_k = 20
#     print("Building NN Graph")
#     start_time = time.time()
#     index = faiss.IndexFlatL2(X.shape[1])
#     index.add(X)
#     distances, indices = index.search(X, k=k + 1)
#     distances, indices = distances[:, 1:], indices[:, 1:]
#     row = np.tile(np.arange(X.shape[0])[:, None], k)

#     # min_distances, min_indices = distances[:, :k], indices[:,:k]
#     # min_row = row = np.tile(np.arange(X.shape[0])[:, None], k)
#     if weighted:
#         nn_graph = csr_matrix(
#             (distances.flatten(), (row.flatten(), indices.flatten())),
#             shape=(X.shape[0], X.shape[0]),
#         )

#         # min_graph = csr_matrix(
#         #     (min_distances.flatten(), (min_row.flatten(), min_indices.flatten())),
#         #     shape=(X.shape[0], X.shape[0]),
#         # )
#     else:
#         nn_graph = csr_matrix(
#             (np.ones(distances.flatten().shape), (row.flatten(), indices.flatten())),
#             shape=(X.shape[0], X.shape[0]),
#         )
#     #     min_graph = csr_matrix(
#     #         (np.ones(min_distances.flatten()), (min_row.flatten(), min_indices.flatten())),
#     #         shape=(X.shape[0], X.shape[0]),
#     #     )

#     print("NN Time: " + str(time.time() - start_time))

#     # # Get minimum spanning tree to ensure full connectivity in graph
#     # start_time = time.time()
#     # min_span_tree = minimum_spanning_tree(nn_graph)
#     # min_span_tree.data = min_span_tree.data.astype(X.dtype)
#     # print("Minimum Spanning Tree Time: " + str(time.time() - start_time))

#     # # Get union between minimum spanning tree and nn graph
#     # min_span_tree_insert = min_span_tree - nn_graph
#     # min_span_tree_insert.data = np.where(min_span_tree_insert.data < 0, 1, 0)
#     # graph = (
#     #     min_span_tree
#     #     - min_span_tree.multiply(min_span_tree_insert)
#     #     + nn_graph.multiply(min_span_tree_insert)
#     # )

#     return nn_graph

def get_pose_geodesic(
    pose: npt.NDArray[Any],
    graph: csr_matrix,
    start_i: int,
    end_i: int,
) -> Tuple[npt.NDArray[np.float64], List[int]]:
    """
    Return the poses along the geodesic path between two frames on a graph.

    Parameters
    ----------
    pose : ndarray, shape (n_frames, n_keypoints, 3)
        3D pose arrays for each frame.
    graph : csr_matrix, shape (n_frames, n_frames)
        Nearest-neighbor graph (weighted) connecting frames.
    start_i : int
        Index of the starting frame.
    end_i : int
        Index of the ending frame.

    Returns
    -------
    geodesic_pose : ndarray(float64), shape (m, n_keypoints, 3)
        Sequence of poses along the geodesic (starts at start_i, ends at end_i).
    indices : list of int
        Frame indices corresponding to rows of `geodesic_pose`.
    """
    print("Calculating Dijkstra")
    path_indices = dijkstra(
        csgraph=graph, directed=False, indices=end_i, return_predecessors=True
    )[1]

    print("Finding pose geodesic")
    geodesic_pose: List[npt.NDArray[Any]] = []
    geodesic_indices: List[int] = []
    curr_frame = start_i

    while path_indices[curr_frame] > 0:
        geodesic_pose += [pose[curr_frame : curr_frame + 1, ...]]
        geodesic_indices += [int(curr_frame)]
        curr_frame = int(path_indices[curr_frame])

    geodesic_pose += [pose[end_i: end_i + 1, ...]]
    geodesic_indices += [int(end_i)]
    if curr_frame != end_i:
        print("Broken graph")

    geodesic_pose = np.concatenate(geodesic_pose, axis=0)

    return np.asarray(geodesic_pose).astype(np.float64), geodesic_indices


def hist_cluster_by_watershed(
    data: npt.NDArray[Any], watershed: Watershed
) -> npt.NDArray[np.float64]:
    """
    Generate normalized histogram of cluster occupancies using a Watershed segmentation.

    Parameters
    ----------
    data : ndarray, shape (n_frames, 2)
        2D embedded coordinates for frames.
    watershed : Watershed
        Watershed segmentation object with precomputed `watershed_map`.

    Returns
    -------
    freq : ndarray(float64), shape (n_clusters,)
        Normalized occupancy histogram across clusters (sums to 1).
    """
    num_clusters = np.max(watershed.watershed_map) + 1
    cluster_labels = watershed.predict(data)

    # Calculate frequencies
    freq = hist_cluster(cluster_labels, num_clusters)
    return np.asarray(freq).astype(np.float64)


def hist_cluster(cluster_labels: npt.NDArray[Any], num_clusters: int) -> npt.NDArray[np.float64]:
    """
    Generate a normalized histogram over cluster labels.

    Parameters
    ----------
    cluster_labels : ndarray, shape (n_frames,)
        Integer cluster label for each frame.
    num_clusters : int
        Number of clusters (max label + 1).

    Returns
    -------
    histogram : ndarray(float64), shape (num_clusters,)
        Normalized histogram (density) over cluster labels.
    """
    freq = np.histogram(
        cluster_labels,
        bins=num_clusters,
        range=(-0.5, num_clusters - 0.5),
        density=True,
    )[0]
    return np.asarray(freq).astype(np.float64)


def hist_cluster_by_cat(
    cluster_labels: npt.NDArray[Any],
    cat: npt.NDArray[Any],
    return_labels: bool = False,
) -> Union[npt.NDArray[np.float64], Tuple[npt.NDArray[np.float64], npt.NDArray[Any]]]:
    """
    Compute histograms of cluster occupancies separated by categorical labels.

    Parameters
    ----------
    cluster_labels : ndarray, shape (n_frames,)
        Cluster label for each frame.
    cat : ndarray, shape (n_frames,)
        Categorical labels per frame (e.g., condition or subject).
    return_labels : bool, default False
        If True, also return the ordered list of unique category labels.

    Returns
    -------
    freq : ndarray(float64), shape (n_categories, n_clusters)
        Frequency histograms per category.
    labels : ndarray, shape (n_categories,), optional
        Unique category labels in the order used for rows (returned if return_labels is True).
    """
    print("Calculating cluster occupancies ")
    num_clusters = int(np.max(cluster_labels)) + 1
    # Unique labels in stable order (first occurrence)
    cat_labels = cat[np.sort(np.unique(cat, return_index=True)[1])]
    freq = np.zeros((len(cat_labels), num_clusters))
    for i, label in enumerate(tqdm(cat_labels)):
        freq[i, :] = hist_cluster(cluster_labels[cat == label], num_clusters)

    if return_labels:
        return np.asarray(freq).astype(np.float64), np.asarray(cat_labels)
    else:
        return np.asarray(freq).astype(np.float64)


def cosine_similarity(a: npt.NDArray[Any], b: npt.NDArray[Any]) -> npt.NDArray[np.float64]:
    """
    Compute row-wise cosine similarity between rows of `a` and rows of `b`.

    Parameters
    ----------
    a : ndarray, shape (n_rows, n_dim)
        First matrix (rows are vectors).
    b : ndarray, shape (m_rows, n_dim)
        Second matrix. If m_rows != n_rows, the result is (n_rows, m_rows).

    Returns
    -------
    cosine_similarity : ndarray(float64), shape (n_rows, m_rows)
        Cosine similarity matrix between rows of `a` and rows of `b`.
    """
    a = np.asarray(a)
    b = np.asarray(b)
    norm_a = np.linalg.norm(a, axis=1)
    norm_b = np.linalg.norm(b, axis=1)
    cos_sim = (a @ b.T) / (norm_a * norm_b)
    return np.asarray(cos_sim).astype(np.float64)


def _bin_embed_distance(
    values: npt.NDArray[Any],
    meta: npt.NDArray[Any],
    augmentation: npt.NDArray[Any],
    time_bins: int = 1000,
    hist_bins: int = 100,
    hist_range: Optional[npt.NDArray[Any]] = None,
) -> npt.NDArray[np.float64]:
    """
    Calculate Jensen-Shannon distance between temporally-binned 2D histograms across augmentations.

    Parameters
    ----------
    values : ndarray, shape (n_samples, n_points, 2)
        2D embedded coordinates for samples (frames x points x 2).
    meta : ndarray, shape (n_samples,)
        Metadata array aligning each sample to an augmentation label.
    augmentation : ndarray, shape (n_augmentations,)
        Ordered augmentation labels to compare (first element is baseline).
    time_bins : int, default 1000
        Number of temporal bins to split each augmentation's frames into.
    hist_bins : int, default 100
        Number of bins per axis for the 2D histogram.
    hist_range : ndarray or None, optional
        Range argument passed to `np.histogram2d` as `[[xmin, xmax], [ymin, ymax]]`.

    Returns
    -------
    dist_js : ndarray(float64), shape (len(augmentation) - 1,)
        Mean Jensen-Shannon distance between baseline histograms and each subsequent augmentation.
    """
    dist_js = np.zeros(len(augmentation) - 1)
    dist_med, dist_mse = np.zeros(len(dist_js)), np.zeros(len(dist_js))
    for i in range(len(augmentation)):
        vals_aug = values[meta == augmentation[i]]
        remainder = vals_aug.shape[0] % time_bins

        if remainder == 0:
            bin_aug = vals_aug.reshape((time_bins, -1, 2))
        else:
            bin_aug = vals_aug[:-remainder, ...].reshape((time_bins, -1, 2))

        stacked_hist = np.empty((0, hist_bins**2))
        for j in range(time_bins):
            stacked_hist = np.append(
                stacked_hist,
                np.histogram2d(
                    bin_aug[j, :, 0],
                    bin_aug[j, :, 1],
                    bins=[hist_bins, hist_bins],
                    range=hist_range,
                    density=True,
                )[0].reshape((1, -1)),
                axis=0,
            )

        if i == 0:
            vals_base = vals_aug
            hist_base = stacked_hist
        else:
            dist_js[i - 1] = np.mean(
                np.array(
                    [
                        distance.jensenshannon(stacked_hist[ii, :], hist_base[ii, :])
                        for ii in range(time_bins)
                    ]
                )
            )

    return np.asarray(dist_js).astype(np.float64)


def levenshtein(
    s1: Union[str, Sequence[Any], npt.NDArray[Any]],
    s2: Union[str, Sequence[Any], npt.NDArray[Any]],
) -> int:
    """Compute the Levenshtein edit distance between two sequences.

    Parameters
    ----------
    s1 : str or sequence, shape (len1,)
        First sequence (string, list, or 1D ndarray).
    s2 : str or sequence, shape (len2,)
        Second sequence.

    Returns
    -------
    distance : int
        Minimum number of insertions, deletions, or substitutions required
        to transform `s1` into `s2`.
    """
    # Normalize to sequences that support len()/indexing
    if len(s1) < len(s2):
        return levenshtein(s2, s1)

    # len(s1) >= len(s2)
    if len(s2) == 0:
        return int(len(s1))

    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return int(previous_row[-1])
