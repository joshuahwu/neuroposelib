import numpy as np
from tqdm import tqdm
from typing import Union, List, Optional
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

def get_z_scores(freq, meta, groups):
    cluster_list = list(np.arange(freq.shape[-1], dtype=int))#list(np.sort(freq_df.columns))
    # if 0 in cluster_list:
    #     cluster_list.pop(0)
    freq_df = pd.DataFrame(freq)
    freq_df.loc[:, 0] = 0
    freq_df = pd.concat([meta.loc[:, groups], freq_df], axis=1)
    # freq_df_BH = freq_df.loc[freq_df["Condition"].isin(["Baseline", "Habituation"]), :]
    # groups = ["Merged_GroupID"]
    meanz = (
        freq_df[cluster_list + groups]
        .groupby(groups)
        .mean()
    )
    stdz = (
        freq_df[cluster_list + groups]
        .groupby(groups)
        .std()
    )
    n_s = freq_df.groupby(groups).size().values
    mean_diffs = np.array(meanz)[:, None, ...] - np.array(meanz)[None, ...]
    std_diffs = np.sqrt(
        np.array(stdz)[:, None, ...] ** 2 / n_s[:, None, None]
        + np.array(stdz)[None, ...] ** 2 / n_s[None, :, None]
    )
    z_scores = mean_diffs / np.where(std_diffs == 0, 1, std_diffs)
    labels = list(meanz.index)

    return z_scores, labels

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
    pose: npt.NDArray,
    graph: csr_matrix,
    start_i: int,
    end_i: int,
) -> tuple[npt.NDArray, List]:
    """Return the poses along the geodesics defined by a nearest neighbor graph.

    Parameters
    ----------
    pose : npt.NDArray
        Array of 3D pose values of shape (# frames, # keypoints, 3 coordinates).
    graph : csr_matrix
        Nearest neighbor graph.
    start_i : int
        Index of first pose.
    end_i : int
        Index of second pose.

    Returns
    -------
    geodesic_pose : npt.NDArray
        Frames of poses along the pose geodesic which begins with `pose[start_i]` and ends with `pose[end_i]`.
    indices: List
        Indices within `pose` corresponding to the pose geodesic.
    """    
    print("Calculating Dijkstra")
    path_indices = dijkstra(
        csgraph=graph, directed=False, indices=end_i, return_predecessors=True
    )[1]

    print("Finding pose geodesic")
    geodesic_pose, geodesic_indices = [], []
    curr_frame = start_i

    while path_indices[curr_frame] > 0:
        geodesic_pose += [pose[curr_frame : curr_frame + 1, ...]]
        geodesic_indices += [curr_frame]
        curr_frame = path_indices[curr_frame]

    geodesic_pose += [pose[end_i: end_i + 1, ...]]
    geodesic_indices += [end_i]
    if curr_frame != end_i:
        print("Broken graph")

    geodesic_pose = np.concatenate(geodesic_pose, axis=0)

    return geodesic_pose, geodesic_indices


def hist_cluster_by_watershed(data: npt.NDArray, watershed: Watershed) -> npt.NDArray:
    """Generates histogram of cluster assignments given 2D embedded values and a Watershed segmentation object.

    Parameters
    ----------
    data : npt.NDArray
        2D embedded values (# frames, 2).
    watershed : Watershed
        Watershed segmentation object.

    Returns
    -------
    histogram: npt.NDArray
        Histogram (# clusters).
    """
    num_clusters = np.max(watershed.watershed_map) + 1
    cluster_labels = watershed.predict(data)

    # Calculate frequencies
    freq = hist_cluster(cluster_labels, num_clusters)
    return freq


def hist_cluster(cluster_labels: npt.NDArray, num_clusters: int) -> npt.NDArray:
    """Generates histograms of cluster assignments.

    Parameters
    ----------
    cluster_labels : npt.NDArray
        Cluster labels per frame (# frames).
    num_clusters : int
        Total number of clusters.

    Returns
    -------
    histogram: npt.NDArray
        Histogram (# clusters).
    """    
    freq = np.histogram(
        cluster_labels,
        bins=num_clusters,
        range=(-0.5, num_clusters - 0.5),
        density=True,
    )[0]
    return freq

def hist_cluster_by_cat(cluster_labels: npt.ArrayLike, cat: npt.ArrayLike, return_labels: bool = False) -> tuple[npt.NDArray, npt.ArrayLike]:
    """Generates histograms of cluster assignments organized by categorical label.

    Parameters
    ----------
    cluster_labels :
        Cluster labels per frame (# frames).
    cat :
        Categorical labels (# frames).

    Returns
    -------
    histogram: npt.NDArray
        Histogram (# categories, # clusters).
    labels : npt.ArrayLike
        If `return_labels == True`, returns unique labels in categories.
    """    
    print("Calculating cluster occupancies ")
    num_clusters = np.max(cluster_labels) + 1
    cat_labels = cat[np.sort(np.unique(cat, return_index=True)[1])]  # Unique cat labels
    freq = np.zeros((len(cat_labels), num_clusters))
    for i, label in enumerate(tqdm(cat_labels)):
        # import pdb; pdb.set_trace()
        freq[i, :] = hist_cluster(
            cluster_labels[cat == label], num_clusters
        )

    if return_labels:
        return freq, cat_labels
    else:
        return freq

def cosine_similarity(a: npt.NDArray, b: npt.NDArray):
    """Row-wise cosine similarity between two 2D matrices. `a` and `b` must match in shape.

    Parameters
    ----------
    a : npt.NDArray
    b : npt.NDArray

    Returns
    -------
    cosine_similarity
        Cosine similarity between each row of a and b.
    """    
    norm_a = np.linalg.norm(a, axis=1)
    norm_b = np.linalg.norm(b, axis=1)
    cos_sim = (a @ b.T) / (norm_a * norm_b)

    return cos_sim

def _bin_embed_distance(
    values: npt.NDArray,
    meta: npt.ArrayLike,
    augmentation: npt.ArrayLike,
    time_bins: int = 1000,
    hist_bins: int = 100,
    hist_range: Optional[npt.NDArray] = None,
):
    """Calculating Jensen Shannon distance between binned segments of videos

    Parameters
    ----------
    values : npt.NDArray
        _description_
    meta : npt.ArrayLike
        _description_
    augmentation : npt.ArrayLike
        _description_
    time_bins : int, optional
        _description_, by default 1000
    hist_bins : int, optional
        _description_, by default 100
    hist_range : Optional[npt.NDArray], optional
        _description_, by default None

    Returns
    -------
    _type_
        _description_
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
            # import pdb; pdb.set_trace()
            dist_js[i - 1] = np.mean(
                np.array(
                    [
                        distance.jensenshannon(stacked_hist[i, :], hist_base[i, :])
                        for i in range(time_bins)
                    ]
                )
            )
            # dist_mse[i-1] = np.sum((vals_base - vals_aug) ** 2) / len(vals_base)
            # dist_med[i-1] = np.sqrt(np.sum((vals_base - vals_aug) ** 2)) / len(vals_base)

    return dist_js  # , dist_mse, dist_med


def levenshtein(s1: npt.ArrayLike, s2: npt.ArrayLike):
    """Levenshtein edit distance between two sequences.

    From [Wikipedia](https://en.wikibooks.org/wiki/Algorithm_Implementation/Strings/Levenshtein_distance#Python).

    Parameters
    ----------
    s1 : npt.ArrayLike
        Sequence 1.
    s2 : npt.ArrayLike
        Sequence 2.

    Returns
    -------
    distance : int
        Number of insertions, deletions, and substitutions to convert `s1` to `s2`.
    """
    if len(s1) < len(s2):
        return levenshtein(s2, s1)

    # len(s1) >= len(s2)
    if len(s2) == 0:
        return len(s1)

    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = (
                previous_row[j + 1] + 1
            )  # j+1 instead of j since previous_row and current_row are one character longer
            deletions = current_row[j] + 1  # than s2
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]