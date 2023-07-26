# from dappy.features import *
# from dappy import DataStruct as ds
# from dappy import visualization as vis
# from dappy import interface as itf
import numpy as np
from tqdm import tqdm
from typing import Union, List, Optional
import sklearn
from sklearn.linear_model import ElasticNet, ElasticNetCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from sklearn.ensemble import RandomForestRegressor
import seaborn as sns
from dappy.embed import Watershed
import faiss
import time
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra, minimum_spanning_tree
from scipy.spatial import distance

def get_nn_graph(X: np.ndarray, k: int = 5, weighted: bool = True):
    X = np.ascontiguousarray(X, dtype=np.float32)

    # max_k = 20
    print("Building NN Graph")
    start_time = time.time()
    index = faiss.IndexFlatL2(X.shape[1])
    index.add(X)
    distances, indices = index.search(X, k=k+1)
    distances, indices = distances[:, 1:], indices[:, 1:]
    row = np.tile(np.arange(X.shape[0])[:, None], k)

    # min_distances, min_indices = distances[:, :k], indices[:,:k]
    # min_row = row = np.tile(np.arange(X.shape[0])[:, None], k)
    if weighted:
        nn_graph = csr_matrix(
            (distances.flatten(), (row.flatten(), indices.flatten())),
            shape=(X.shape[0], X.shape[0]),
        )

        # min_graph = csr_matrix(
        #     (min_distances.flatten(), (min_row.flatten(), min_indices.flatten())),
        #     shape=(X.shape[0], X.shape[0]),
        # )
    else:
        nn_graph = csr_matrix(
            (np.ones(distances.flatten().shape), (row.flatten(), indices.flatten())),
            shape=(X.shape[0], X.shape[0]),
        )
    #     min_graph = csr_matrix(
    #         (np.ones(min_distances.flatten()), (min_row.flatten(), min_indices.flatten())),
    #         shape=(X.shape[0], X.shape[0]),
    #     )

    print("NN Time: " + str(time.time() - start_time))

    # # Get minimum spanning tree to ensure full connectivity in graph
    # start_time = time.time()
    # min_span_tree = minimum_spanning_tree(nn_graph)
    # min_span_tree.data = min_span_tree.data.astype(X.dtype)
    # print("Minimum Spanning Tree Time: " + str(time.time() - start_time))

    # # Get union between minimum spanning tree and nn graph
    # min_span_tree_insert = min_span_tree - nn_graph
    # min_span_tree_insert.data = np.where(min_span_tree_insert.data < 0, 1, 0)
    # graph = (
    #     min_span_tree
    #     - min_span_tree.multiply(min_span_tree_insert)
    #     + nn_graph.multiply(min_span_tree_insert)
    # )

    return nn_graph


def get_pose_geodesic(
    pose: np.ndarray,
    graph: csr_matrix,
    START_FRAME: int,
    END_FRAME: int,
):
    print("Calculating Dijkstra")
    path_indices = dijkstra(
        csgraph=graph, directed=False, indices=END_FRAME, return_predecessors=True
    )[1]

    print("Finding pose geodesic")
    geodesic_pose, geodesic_indices = [], []
    curr_frame = START_FRAME

    while path_indices[curr_frame] > 0:
        geodesic_pose += [pose[curr_frame : curr_frame + 1, ...]]
        geodesic_indices += [curr_frame]
        curr_frame = path_indices[curr_frame]

    if curr_frame != END_FRAME:
        print("Broken graph")

    geodesic_pose = np.concatenate(geodesic_pose, axis=0)

    return geodesic_pose, geodesic_indices

def cluster_freq_from_data(data: np.ndarray, watershed: Watershed):
    """
    Gets the cluster frequency of the data
    IN:
        data: embedded values of data (# frames x 2)
        watershed: Watershed object
    """
    num_clusters = np.max(watershed.watershed_map) + 1
    cluster_labels = watershed.predict(data)

    # Calculate frequencies
    freq = cluster_freq_from_labels(cluster_labels, num_clusters)

    return freq


def cluster_freq_from_labels(cluster_labels: np.ndarray, num_clusters: int):
    freq = np.histogram(
        cluster_labels,
        bins=num_clusters,
        range=(-0.5, num_clusters - 0.5),
        density=True,
    )[0]
    return freq


def cluster_freq_by_cat(cluster_labels: np.ndarray, cat: np.ndarray):
    print("Calculating cluster occupancies ")
    num_clusters = np.max(cluster_labels) + 1
    cat_labels = cat[np.sort(np.unique(cat, return_index=True)[1])]  # Unique cat labels
    freq = np.zeros((len(cat_labels), num_clusters))
    for i, label in enumerate(tqdm(cat_labels)):
        # import pdb; pdb.set_trace()
        freq[i, :] = cluster_freq_from_labels(
            cluster_labels[cat == label], num_clusters
        )

    return freq, cat_labels


def lstsq(freq: np.ndarray, y: np.ndarray, filepath: str):
    print("Applying Least Squares Regression")
    pred_y = np.zeros(y.shape)
    for i in range(len(y)):
        m = np.linalg.lstsq(np.delete(freq, i, axis=0), np.delete(y, i))[0]
        pred_y[i] = freq[i, :] @ m

    print("R2 Score " + str(r2_score(y, pred_y)))
    return pred_y


def elastic_net(freq: np.ndarray, y: np.ndarray, filepath: str):
    print("Applying ElasticNet Regression")
    pred_y = np.zeros(y.shape)
    for i in range(len(y)):
        regr = ElasticNet(alpha=0.1, l1_ratio=0.7)

        temp_lesion = np.delete(freq, i, axis=0)
        scaler = StandardScaler().fit(temp_lesion)

        regr.fit(scaler.transform(temp_lesion), np.log2(np.delete(y, i)))
        pred_y[i] = regr.predict(scaler.transform(freq[i, :][None, :]))

    # sns.set(rc={'figure.figsize':(6,5)})
    # f = plt.figure()
    # # import pdb; pdb.set_trace()
    # plt.plot(np.linspace(y.min(), y.max(), 100), np.linspace(y.min(),y.max(),100), markersize=0, color='k', label="y = x")
    # plt.legend(loc="upper center")
    # plt.scatter(y, 2**pred_y, s=30)
    # plt.xlabel("Real Fluorescence")
    # plt.ylabel("Predicted Fluorescence")
    # plt.savefig("".join([filepath, "elastic.png"]))
    # plt.close()

    print("R2 Score " + str(r2_score(y, 2**pred_y)))
    return pred_y, 

def elastic_net_cv(freq: np.ndarray, y: np.ndarray, filepath: str):
    print("Applying ElasticNet Regression")
    pred_y = np.zeros(y.shape)
    # pred_y2 = np.zeros(y.shape)
    for i in tqdm(range(len(y))):
        # Predict single from the rest
        regr = ElasticNetCV(n_alphas=50, l1_ratio=[.1, .5, .7, .9, .95, .99, 1],
                            cv=10)

        temp_lesion = np.delete(freq, i, axis=0)
        scaler = StandardScaler().fit(temp_lesion)

        regr.fit(scaler.transform(temp_lesion), np.delete(y, i))
        pred_y[i] = regr.predict(scaler.transform(freq[i, :][None, :]))
         #TODO: Check convergence issue
         #TODO: Try dropping PDb8 from r^2 calculation
        # pred_y2[i] = regr.predict(scaler.transform(freq2[i,:][None, :]))

    # sns.set(rc={'figure.figsize':(6,5)})
    # f = plt.figure()
    # # import pdb; pdb.set_trace()
    # plt.plot(np.linspace(y.min(), y.max(), 100), np.linspace(y.min(),y.max(),100), markersize=0, color='k', label="y = x")
    # plt.legend(loc="upper center")
    # plt.scatter(y, pred_y, s=30)
    # plt.xlabel("Real Fluorescence")
    # plt.ylabel("Predicted Fluorescence")
    # plt.savefig("".join([filepath, "elastic.png"]))
    # plt.close()

    # plt.plot(np.linspace(y.min(), y.max(), 100), np.linspace(y.min(),y.max(),100), markersize=0, color='k', label="y = x")
    # plt.legend(loc="upper center")
    # plt.scatter(y, pred_y2, s=30)
    # plt.xlabel("Real Fluorescence")
    # plt.ylabel("Predicted Fluorescence")
    # plt.savefig("".join([filepath, "elastic_healthy_same_coeff.png"]))
    # plt.close()

    print("R2 Score " + str(r2_score(y, pred_y)))
    return pred_y, r2_score(y,pred_y)


def random_forest(freq: np.ndarray, y: np.ndarray, filepath: str):
    print("Applying Random Forest Regression")
    pred_y = np.zeros(y.shape)
    for i in range(len(y)):
        rf_regr = RandomForestRegressor()
        rf_regr.fit(np.delete(freq, i, axis=0), np.delete(y, i))
        pred_y[i] = rf_regr.predict(freq[i, :][None, :])

    # plt.scatter(y, pred_y)
    # plt.xlabel("Real Fluorescence")
    # plt.ylabel("Predicted Fluorescence")
    # plt.savefig("".join([filepath, "rforest.png"]))
    # plt.close()
    print("R2 Score " + str(r2_score(y, pred_y)))
    return


def pairwise_cosine(cluster_freq: np.ndarray, filepath: str):
    paired_cosine = sklearn.metrics.pairwise.cosine_similarity(cluster_freq)
    paired_cosine = np.delete(paired_cosine,[30,67],axis=0)
    paired_cosine = np.delete(paired_cosine,[30,67],axis=1)
    num_subjects = int(paired_cosine.shape[0] / 2)

    labels = ["B " + str(i) for i in range(num_subjects)]
    labels += ["L " + str(i) for i in range(num_subjects)]
    # pair_cos_df = pd.DataFrame(paired_cosine, index = labels, columns = labels)
    # sns.set(rc={'figure.figsize':(12,10)})
    # ax = sns.heatmap(pair_cos_df,cmap = sns.color_palette("mako",as_cmap=True))
    # ax.set_aspect('equal','box')
    # ax.figure.savefig("".join([filepath,"pairwise_cosine.png"]))
    # plt.close()
    palette = ["#00b7c7", "#dc0ab4"]
    tri_ind = np.triu_indices(num_subjects, 1)

    sns.set(rc={'figure.figsize':(6,5)})
    cond_1 = paired_cosine[:num_subjects, :num_subjects][tri_ind]
    cond_2 = paired_cosine[num_subjects:, num_subjects:][tri_ind]

    data = np.append(cond_1, cond_2)
    labels = np.empty(data.shape, dtype=object)
    labels[: len(cond_1)] = "Baseline"
    labels[len(cond_1) :] = "Lesion"
    inner_cos_df = pd.DataFrame(data, columns=["Pairwise Cosine Similarity"])
    inner_cos_df["Condition"] = labels
    ax = sns.catplot(
        data=inner_cos_df,
        y="Pairwise Cosine Similarity",
        x="Condition",
        kind="violin",
        errorbar="se",
        palette=palette,
        alpha=0.1
    )

    ax.map_dataframe(sns.stripplot,x="Condition", y="Pairwise Cosine Similarity", palette=["#404040"], s = 2,
                alpha=0.6, jitter=0.3)
    ax.figure.savefig("".join([filepath, "pair_cos_violin.png"]))
    plt.close()

    return paired_cosine


def cosine_similarity(a: np.ndarray, b: np.ndarray):
    """
    Row-wise cosine similarity between two 2D matrices
    """
    norm_a = np.linalg.norm(a, axis=1)
    norm_b = np.linalg.norm(b, axis=1)
    # import pdb; pdb.set_trace()
    cos_sim = np.sum(a * b, axis=1) / (norm_a * norm_b)

    return cos_sim


## Calculating Jensen Shannon distance between binned segments of videos
def bin_embed_distance(
    values: np.ndarray,
    meta: Union[np.ndarray, List],
    augmentation: Union[np.ndarray, List],
    time_bins: int = 1000,
    hist_bins: int = 100,
    hist_range: Optional[np.ndarray] = None,
):
    dist_js = np.zeros(len(augmentation)-1)
    dist_med, dist_mse = np.zeros(len(dist_js)), np.zeros(len(dist_js))
    for i in range(len(augmentation)):
        vals_aug = values[meta == augmentation[i]]
        remainder = vals_aug.shape[0] % time_bins

        if remainder == 0:
            bin_aug = vals_aug.reshape((time_bins, -1, 2))
        else:
            bin_aug = vals_aug[:-remainder,...].reshape((time_bins, -1, 2))

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
            dist_js[i-1] = np.mean(
                np.array(
                    [
                        distance.jensenshannon(stacked_hist[i, :], hist_base[i, :])
                        for i in range(time_bins)
                    ]
                )
            )
            # dist_mse[i-1] = np.sum((vals_base - vals_aug) ** 2) / len(vals_base)
            # dist_med[i-1] = np.sqrt(np.sum((vals_base - vals_aug) ** 2)) / len(vals_base)

    return dist_js#, dist_mse, dist_med

def levenshtein(s1, s2):
    '''
    From https://en.wikibooks.org/wiki/Algorithm_Implementation/Strings/Levenshtein_distance#Python
    '''
    if len(s1) < len(s2):
        return levenshtein(s2, s1)

    # len(s1) >= len(s2)
    if len(s2) == 0:
        return len(s1)

    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1 # j+1 instead of j since previous_row and current_row are one character longer
            deletions = current_row[j] + 1       # than s2
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    
    return previous_row[-1]

