import numpy as np
import tqdm

import pandas as pd
# import seaborn as sns
from matplotlib.lines import Line2D
import matplotlib
from pathlib import Path

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from typing import Optional, Union, List
from scipy.special import softmax

from neuroposelib import DataStruct as ds
from neuroposelib.embed import Watershed, GaussDensity
from neuroposelib.analysis import hist_cluster_by_cat
from neuroposelib.visualization.constants import PALETTE, EPS, DEFAULT_VIRIDIS


# def scatter_by_cat(
#     data: np.ndarray,
#     cat: np.ndarray,
#     label: str,
#     size=3,
#     color=None,
#     filepath: str = "./",
# ):
#     if color == None:
#         color = PALETTE
#     ax = sns.scatterplot(
#         x=data[:, 0],
#         y=data[:, 1],
#         marker=".",
#         hue=cat,
#         palette=color,
#         s=size,
#     )
#     # ax.set(xlabel="t-SNE 1", ylabel="t-SNE 2")
#     ax.set_box_aspect(0.9)
#     ax.figure.savefig("".join([filepath, "scatter_by_", label, ".png"]))
#     plt.close()
#     # plt.savefig(''.join([filepath, 'scatter_by_', label, '.png']))


def scatter(
    data: np.ndarray,
    color: Optional[Union[List, np.ndarray]] = None,
    marker_size: int = 3,
    ax_label: str = "t-SNE",
    filepath: str = "./results/scatter.png",
    show: bool = False,
    **kwargs,
):
    """
    Draw a 2d tSNE plot from zValues.

    input: zValues dataframe, [num of points x 2]
    output: a scatter plot
    """
    if data.shape.index(2) == 1:
        x = data[:, 0]
        y = data[:, 1]
    else:
        x = data[0, :]
        y = data[1, :]

    f = plt.figure()
    plt.scatter(
        x,
        y,
        marker=".",
        s=marker_size,
        linewidths=0,
        c=color,
        # cmap=sns.color_palette("crest", as_cmap=True),
        alpha=0.75,
        **kwargs,
    )
    plt.xlabel(ax_label + " 1")
    plt.ylabel(ax_label + " 2")
    if color is not None:
        plt.colorbar()
    if filepath:
        plt.savefig(filepath, dpi=200)

    if show:
        plt.show()
    plt.close()


def watershed(
    ws_map: np.ndarray,
    ws_borders: Optional[np.ndarray] = None,
    cmap: Optional[str] = None,
    filepath: str = "./results/watershed.png",
):
    """
    Plotting a watershed map with clusters colored

    """
    if cmap == None:
        cmap = DEFAULT_VIRIDIS
    f = plt.figure()
    ax = f.add_subplot(111)
    ax.imshow(ws_map, vmin=EPS, cmap=cmap)
    ax.set_aspect(0.9)
    if ws_borders is not None:
        ax.plot(ws_borders[:, 0], ws_borders[:, 1], ".k", markersize=0.05)

    ax.axis("off")
    plt.savefig("".join([filepath, "_watershed.png"]), dpi=200)
    plt.close()


def scatter_on_watershed(
    data: ds.DataStruct, watershed: GaussDensity, column: str, path: str = "./results/"
):
    out_path = "{}points_by_{}/".format(path, column)
    labels = data.data[column].values
    Path(out_path).mkdir(parents=True, exist_ok=True)
    extent = [*watershed.hist_range[0], *watershed.hist_range[1]]

    f = plt.figure()
    ax = f.add_subplot(111)
    ax.imshow(
        watershed.watershed_map,
        zorder=1,
        extent=extent,
        vmin=EPS,
        cmap=DEFAULT_VIRIDIS,
    )
    ax.plot(
        data.embed_vals[:, 0],
        data.embed_vals[:, 1],
        marker=".",
        c="k",
        markersize=1,
        alpha=0.1,
        zorder=2,
    )
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect(0.9)
    ax.axis("off")
    filename = "{}all.png".format(out_path)
    plt.savefig(filename, dpi=200)
    plt.close()

    print("Plotting scatter on watershed for each ", column)
    for i, label in enumerate(tqdm.tqdm(np.unique(labels))):
        embed_vals = data.embed_vals[data.data[column] == label]

        f = plt.figure()
        ax = f.add_subplot(111)
        ax.imshow(
            watershed.watershed_map,
            zorder=0,
            extent=extent,
            vmin=EPS,
            cmap=DEFAULT_VIRIDIS,
        )

        ax.plot(
            embed_vals[:, 0],
            embed_vals[:, 1],
            marker=".",
            c="k",
            markersize=2,
            alpha=0.1,
            zorder=2,
        )
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect(0.9)
        filename = "{}{}_points_{}.png".format(out_path, column, str(label))
        ax.axis("off")
        plt.savefig(filename, dpi=400)
        plt.close()


def density_feat(
    data: ds.DataStruct,
    watershed: Watershed,
    features: np.ndarray,
    feature_labels: List,
    key: str,
    file_path: str = "./results/",
):
    feat_key = features[:, feature_labels.index(key)]
    density_feat = np.zeros((watershed.n_bins, watershed.n_bins))
    data_in_bin = watershed.map_bins(data.embed_vals)
    min_feat = np.min(feat_key)
    for i in tqdm.tqdm(range(watershed.n_bins)):
        for j in range(watershed.n_bins):
            bin_idx = np.logical_and(data_in_bin[:, 0] == i, data_in_bin[:, 1] == j)

            if np.all(bin_idx == False):
                density_feat[i, j] = min_feat
            else:
                density_feat[i, j] = np.mean(feat_key[bin_idx])

    density(
        density_feat,
        ws_borders=watershed.borders,
        filepath="".join([file_path, "density_feat_", key, ".png"]),
        show=True,
    )


def density(
    density: np.ndarray,
    ws_borders: Optional[np.ndarray] = None,
    filepath: str = "./results/density.png",
    show: bool = False,
    vmax: float = None,
):
    vmin = 0.99 * 15/density.shape[0]**2
    f = plt.figure()
    ax = f.add_subplot(111)
    if ws_borders is not None:
        ax.plot(ws_borders[:, 0], ws_borders[:, 1], ".k", markersize=0.1)

    ax.imshow(density, vmin=vmin, vmax=vmax, cmap=DEFAULT_VIRIDIS)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect(0.9)
    ax.axis("off")
    if filepath:
        plt.savefig(filepath, dpi=200)
    if show:
        plt.show()
    plt.close()


def _mask_density(density, watershed_map, eps: float = EPS * 1.01):
    mask = watershed_map > 0
    density[mask] = np.maximum(density[mask], eps)
    density[~mask] = 0
    return density

def density_cat(
    data: ds.DataStruct,
    column: str,
    watershed: Watershed,
    filepath: str = "./results/density_by_label.png",
    show: bool = False,
    vmax: float = None,
):
    """
    Plot densities by a category label
    """
    labels = data.data[column].values

    n_ulabels = len(np.unique(labels))
    n_rows = int(np.sqrt(n_ulabels))
    n_cols = int(np.ceil(n_ulabels / n_rows))
    f, ax_arr = plt.subplots(n_rows, n_cols, figsize=(n_cols * 4, n_rows * 4))

    # Loop over unique labels
    for i, (label, ax) in enumerate(
        zip(np.unique(labels), ax_arr.reshape(-1)[:n_ulabels])
    ):
        embed_vals = data.embed_vals[data.data[column] == label]  # Indexing by label
        density = watershed.fit_density(
            embed_vals, new=False
        )  # Fit density on old axes

        ax.imshow(
            _mask_density(density, watershed.watershed_map, EPS * 1.01),
            vmin=EPS,
            vmax=vmax,
            cmap=DEFAULT_VIRIDIS,
        )  # scp.special.softmax(density))

        if watershed is not None:
            ax.plot(
                watershed.borders[:, 0],
                watershed.borders[:, 1],
                ".k",
                markersize=0.1,
            )
        ax.set_aspect(0.9)
        ax.set_title(label)

    for ax in ax_arr.reshape(-1):
        ax.set_xticks([])
        ax.set_yticks([])
        ax.axis("off")

    # ax_arr = ax_arr.reshape(n_rows,n_cols)
    f.tight_layout()
    plt.savefig(filepath, dpi=200)
    if show:
        plt.show()
    plt.close()
    return


def density_grid(
    data: ds.DataStruct,
    cat1: str,
    cat2: str,
    watershed: Watershed,
    filepath: str = "./results/density_by_label.png",
    show: bool = False,
    vmax: float = 3.5,
):
    """
    Plot densities by a category label
    """
    labels1, labels2 = data.data[cat1].values, data.data[cat2].values
    n_col = len(np.unique(labels2))
    n_rows = len(np.unique(labels1))
    f, ax_arr = plt.subplots(n_rows, n_col, figsize=((n_col + 1) * 4, n_rows * 4))

    # Loop over unique labels
    for i, label1 in enumerate(np.unique(labels1)):
        # if n_rows != 1:
        #     ax_arr[i, 0].set_title(label1)

        for j, label2 in enumerate(np.unique(labels2)):
            embed_vals = data.embed_vals[
                (data.data[cat1] == label1) & (data.data[cat2] == label2)
            ]  # Indexing by label
            density = watershed.fit_density(
                embed_vals, new=False
            )  # Fit density on old axes
            idx = i * len(np.unique(labels2)) + j
            # if n_rows == 1:
            ax_arr[idx].imshow(
                _mask_density(density, watershed.watershed_map, EPS * 1.01),
                vmin=EPS,
                vmax=vmax,
                cmap=DEFAULT_VIRIDIS,
            )

            if watershed is not None:
                ax_arr[idx].plot(
                    watershed.borders[:, 0],
                    watershed.borders[:, 1],
                    ".k",
                    markersize=0.1,
                )
            ax_arr[idx].set_aspect(0.9)
            # ax_arr[idx].axis("off")
            ax_arr[idx].set_xticks([])
            ax_arr[idx].set_yticks([])
            for spine in ax_arr[idx].spines.values():
                spine.set_visible(False)

            if j == 0:
                ax_arr[idx].set_ylabel(label1)

            if i == 0:
                ax_arr[idx].set_title(label2)

    f.tight_layout()
    plt.savefig(filepath, dpi=200)
    if show:
        plt.show()
    plt.close()
    return


def cluster_freq(data_obj: ds.DataStruct, cat1, cat2, filepath="./", show=False):
    """
    IN:
        cat1: Will create a separate subplot for each label in cat1
        cat2: For each subplot of cat1, will have multiple lines for each cat2
    """
    colors = [
        "tab:green",
        "tab:blue",
        "tab:orange",
        "tab:red",
        "tab:purple",
        "tab:brown",
        "tab:pink",
        "tab:gray",
        "tab:olive",
        "tab:cyan",
        "#dc0ab4",
        "#00b7c7",
    ]
    # Cat1 and cat2 labels for all points
    cat1_labels = data_obj.data[cat1].astype(str).values.tolist()
    cat2_labels = data_obj.data[cat2].astype(str).values.tolist()

    combined_labels = np.array(
        ["_".join([label1, label2]) for label1, label2 in zip(cat1_labels, cat2_labels)]
    )

    freq, combined_keys = hist_cluster_by_cat(
        data_obj.data["Cluster"].values, cat=combined_labels
    )

    # cat1_freq_keys,cat2_freq_keys = map(list, zip(*[key.split(',') for key in combined_keys]))

    num_clusters = freq.shape[1]  # data_obj.data['Cluster'].max()+1
    # Unique keys for cat1 and cat2
    cat1_keys, cat2_keys = np.unique(data_obj.data[cat1].values), np.unique(
        data_obj.data[cat2].values
    )

    f, ax_arr = plt.subplots(len(cat1_keys) + 1, 1, sharex="all", figsize=(20, 10))

    for i, key1 in enumerate(cat1_keys):  # For each plot of cat1
        # videos = data_obj.meta.index[data_obj.meta['Condition'] == cat1_keys[i]].tolist()
        for j, key2 in enumerate(cat2_keys):  # For each cat2 of cat 1
            # cluster_labels = data_obj.data['Cluster'].values[(data_obj.data[cat1]==key1) & data_obj.data[cat2]==key2]
            # freq = analysis.cluster_freq_from_labels(cluster_labels, num_clusters)
            freq_key = "_".join([key1, key2])
            ax_arr[i].plot(
                range(num_clusters),
                np.squeeze(freq[combined_keys == freq_key, :]),
                label=key2,
            )  # color=colors[j], #''.join(['Animal',str(i)]))

        ax_arr[i].set_title(cat1_keys[i], pad=-14)
        ax_arr[i].spines["top"].set_visible(False)
        ax_arr[i].get_xaxis().set_visible(False)
        ax_arr[i].spines["right"].set_visible(False)
        ax_arr[i].spines["bottom"].set_visible(False)
    ax_arr[1].set_ylabel("% Time Spent in Cluster")
    ax_arr[0].legend(loc="upper right", ncol=6)  # TODO: Make this ncol programmable

    markers = ["o", "v", "s"]
    # for j in range(len(conditions),len(conditions)+2): # For mean and variance plots
    j = len(cat1_keys)
    for i, key1 in enumerate(cat1_keys):  # for each condition
        # videos = data.meta.index[data.meta['Condition'] == conditions[j]].tolist()
        key_bool = [True if key.startswith(key1) else False for key in combined_keys]
        ax_arr[j].errorbar(
            range(num_clusters),
            np.mean(freq[key_bool, :], axis=0),
            color=colors[i],
            label=cat1_keys[i],
            marker=markers[i],
            markersize=5,
            linewidth=0,
            elinewidth=1,
            yerr=np.std(freq[key_bool, :], axis=0)
            / np.sqrt(freq[key_bool, :].shape[0]),
        )
        # ax_arr[4].plot(range(num_clusters),np.std(freq[j,:,:],axis=1),color=colors[j],label=conditions[j],
        #                 marker=markers[j], markersize=5, linewidth=0)
    ax_arr[j].spines["top"].set_visible(False)
    ax_arr[j].spines["right"].set_visible(False)

    ax_arr[j].legend(loc="upper right", ncol=3)
    # ax_arr[j].spines['bottom'].set_visible(False)
    # ax_arr[j].get_xaxis().set_visible(False)
    # ax_arr[j].set_ylabel("Mean")
    ax_arr[j].set_ylabel("Mean")
    ax_arr[j].set_xlabel("Cluster")
    ax_arr[j].set_xlim([-0.25, freq.shape[1] + 0.25])
    f.tight_layout()
    plt.subplots_adjust(hspace=0.05)
    plt.savefig("".join([filepath, "mean_sd_cluster_freq.png"]), dpi=200)
    if show:
        plt.show()

    plt.close()
    return


def cluster_freq_cond(data_obj: ds.DataStruct, cat1, cat2, filepath="./", show=False):
    """
    IN:
        cat1: Will create a separate subplot for each label in cat1
        cat2: For each subplot of cat1, will have multiple lines for each cat2
    """
    colors = [
        "tab:pink",
        "tab:green",
        "tab:blue",
        "tab:orange",
        "tab:red",
        "tab:purple",
        "tab:brown",
        "tab:gray",
        "tab:olive",
        "tab:cyan",
        "#dc0ab4",
        "#00b7c7",
    ]
    # Cat1 and cat2 labels for all points
    cat1_labels = data_obj.data[cat1].astype(str).values.tolist()
    cat2_labels = data_obj.data[cat2].astype(str).values.tolist()

    combined_labels = np.array(
        ["_".join([label1, label2]) for label1, label2 in zip(cat1_labels, cat2_labels)]
    )

    freq, combined_keys = hist_cluster_by_cat(
        data_obj.data["Cluster"].values, cat=combined_labels
    )

    # cat1_freq_keys,cat2_freq_keys = map(list, zip(*[key.split(',') for key in combined_keys]))

    num_clusters = freq.shape[1]  # data_obj.data['Cluster'].max()+1
    # Unique keys for cat1 and cat2
    cat1_keys, cat2_keys = np.unique(data_obj.data[cat1].values), np.unique(
        data_obj.data[cat2].values
    )

    f, ax_arr = plt.subplots(
        3, 1, sharex="all", figsize=(12, 4), gridspec_kw={"height_ratios": [0.2, 2, 2]}
    )

    for i, key1 in enumerate(cat1_keys):  # For each plot of cat1
        # videos = data_obj.meta.index[data_obj.meta['Condition'] == cat1_keys[i]].tolist()
        for j, key2 in enumerate(cat2_keys):  # For each cat2 of cat 1
            # cluster_labels = data_obj.data['Cluster'].values[(data_obj.data[cat1]==key1) & data_obj.data[cat2]==key2]
            # freq = analysis.cluster_freq_from_labels(cluster_labels, num_clusters)
            freq_key = "_".join([key1, key2])
            ax_arr[1].plot(
                range(num_clusters),
                np.squeeze(freq[combined_keys == freq_key, :]),
                label=key1,
                color=colors[-i - 1],
                alpha=0.1,
            )  #''.join(['Animal',str(i)]))

    handles = [
        Line2D([0], [0], linewidth=5, color=colors[-i - 1], label=cat1_keys[i])
        for i in range(len(cat1_keys))
    ]
    # ax_arr[0].set_title(cat1_keys[i],pad=-14)
    ax_arr[1].spines["top"].set_visible(False)
    ax_arr[1].get_xaxis().set_visible(False)
    ax_arr[1].spines["right"].set_visible(False)
    ax_arr[1].spines["bottom"].set_visible(False)
    ax_arr[1].set_ylabel("% Time Spent in Cluster")
    ax_arr[1].legend(
        handles, cat1_keys, loc="upper right", ncol=1, borderpad=1
    )  # TODO: Make this ncol programmable
    ax_arr[1].set_ylim(0, 0.16)
    ax_arr[0].set_ylim(0.59, 0.6)
    ax_arr[1].spines["bottom"].set_visible(False)
    ax_arr[0].plot(range(num_clusters), freq[67, :], color=colors[0], alpha=0.1)
    ax_arr[0].get_xaxis().set_visible(False)
    ax_arr[0].spines["bottom"].set_visible(False)
    ax_arr[0].spines["right"].set_visible(False)
    ax_arr[0].spines["top"].set_visible(False)

    d = 4 / 4.2  # proportion of vertical to horizontal extent of the slanted line
    kwargs = dict(
        marker=[(-1, -d), (1, d)],
        markersize=12,
        linestyle="none",
        color="k",
        mec="k",
        mew=1,
        clip_on=False,
    )
    ax_arr[0].plot([0], [0], transform=ax_arr[0].transAxes, **kwargs)
    ax_arr[1].plot([0], [1], transform=ax_arr[1].transAxes, **kwargs)
    markers = ["o", "v", "s"]
    # for j in range(len(conditions),len(conditions)+2): # For mean and variance plots
    j = 1
    for i, key1 in enumerate(cat1_keys):  # for each condition
        # videos = data.meta.index[data.meta['Condition'] == conditions[j]].tolist()
        key_bool = [True if key.startswith(key1) else False for key in combined_keys]

        ax_arr[2].errorbar(
            range(num_clusters),
            np.mean(freq[key_bool, :], axis=0),
            color=colors[-i - 1],
            label=cat1_keys[i],
            marker=markers[i],
            markersize=5,
            linewidth=0,
            elinewidth=1,
            yerr=np.std(freq[key_bool, :], axis=0)
            / np.sqrt(freq[key_bool, :].shape[0]),
        )
        # ax_arr[4].plot(range(num_clusters),np.std(freq[j,:,:],axis=1),color=colors[j],label=conditions[j],
        #                 marker=markers[j], markersize=5, linewidth=0)
    ax_arr[2].spines["top"].set_visible(False)
    ax_arr[2].spines["right"].set_visible(False)

    # ax_arr[1].legend(loc='upper right',ncol=3)
    # ax_arr[j].spines['bottom'].set_visible(False)
    # ax_arr[j].get_xaxis().set_visible(False)
    # ax_arr[j].set_ylabel("Mean")
    ax_arr[2].set_ylabel("Mean")
    ax_arr[2].set_xlabel("Cluster")
    ax_arr[2].set_xlim([-0.25, freq.shape[1] + 0.25])
    f.tight_layout()
    plt.subplots_adjust(hspace=0.05)
    plt.savefig("".join([filepath, "mean_sd_cluster_freq.png"]), dpi=200)
    if show:
        plt.show()

    plt.close()
    return


def heuristics(features, labels, data_obj, heuristics, filepath):
    filepath = filepath + "/heuristics/"
    for heur_key in heuristics:
        print("Plotting heuristics")
        heur_feats = heuristics[heur_key]
        high_feat_i = [
            labels.index(heur_label)
            for heur_label in heur_feats["high"]
            if heur_label in labels
        ]
        low_feat_i = [
            labels.index(heur_label)
            for heur_label in heur_feats["low"]
            if heur_label in labels
        ]

        try:
            assert len(high_feat_i) == len(heur_feats["high"])
            assert len(low_feat_i) == len(heur_feats["low"])
        except:
            print("Couldn't find some features from the heuristics")

        high_feats = np.clip(features[:, high_feat_i], -2.5, 2.5)
        low_feats = np.clip(-features[:, low_feat_i], -2.5, 2.5)
        heur_feats = np.mean(np.append(high_feats, low_feats, axis=1), axis=1)

        # num_clusters = np.max(np.unique(data_obj.data['Cluster'].values))
        heur_watershed = data_obj.ws.watershed_map
        for cluster in np.unique(data_obj.data["Cluster"].values):
            cluster_mean = np.mean(heur_feats[data_obj.data["Cluster"] == cluster])
            heur_watershed[data_obj.ws.watershed_map == cluster] = cluster_mean

        watershed(
            ws_map=heur_watershed,
            ws_borders=data_obj.ws.borders,
            filepath=filepath + heur_key,
        )

        # vis.scatter(data_obj.embed_vals,
        #             color=heur_feats,
        #             filepath=''.join([filepath,'scatter_',heur_key,'_clipped_score.png']))

        print("Highest " + heur_key + " score frames: ")
        sorted_idx = np.argsort(heur_feats)
        print(sorted_idx)

        # skeleton_vid3D(
        #     data_obj.pose,
        #     data_obj.connectivity,
        #     frames=[sorted_idx[-1] * 10],
        #     N_FRAMES=100,
        #     dpi=100,
        #     VID_NAME="highest_" + heur_key + "_score.mp4",
        #     SAVE_ROOT=filepath,
        # )


# def labeled_watershed(watershed, borders, labels_path):
#     labels = pd.read_csv(labels_path)

#     behavior_cats = set(labels["Label"])

#     color_dict = {
#         "Background": np.array([0, 0, 0]),
#         "Crouched Idle": np.array([241, 194, 50]),
#         "Face Groom": np.array([106, 168, 79]),
#         "Body Groom": np.array([106, 168, 79]),
#         "High Rear": np.array([69, 129, 142]),
#         "Low Rear": np.array([106, 168, 79]),
#         "Investigate": np.array([106, 168, 79]),
#         "Walk/Rear": np.array([106, 168, 79]),
#         "Walk": np.array([103, 78, 167]),
#     }

#     colors = [
#         "#FFFFFF",
#         "tab:gray",
#         "tab:green",
#         "tab:blue",
#         "tab:orange",
#         "tab:red",
#         "tab:purple",
#         "tab:brown",
#         "tab:pink",
#         "tab:olive",
#         "tab:cyan",
#         "#dc0ab4",
#         "#00b7c7",
#     ]

#     unique_labels = labels["Label"].unique()

#     labeled_map = np.zeros(watershed.shape)
#     for i, label in enumerate(unique_labels):
#         labeled_map[
#             np.isin(watershed, labels["Cluster"][labels["Label"] == label].values)
#         ] = i

#     # labeled_map = np.zeros((watershed.shape[0], watershed.shape[1], 3))
#     # for i in range(watershed.shape[0]):
#     #     for j in range(watershed.shape[1]):
#     #         # try:
#     #         label = labels["Label"].loc[labels["Cluster"] == watershed[i, j]].values[0]
#     #         # except:
#     #         #     import pdb; pdb.set_trace()
#     #         labeled_map[i, j, :] = np.array(color_dict[label])

#     # fig, ax = plt.subplots()
#     sns.set(rc={"figure.figsize": (12, 10)})
#     cmap = [(1, 1, 1)] + sns.color_palette("Pastel2", len(unique_labels) - 1)
#     ax = sns.heatmap(labeled_map, cmap=cmap)
#     plt.colorbar(ax=ax, fraction=0.047 * 0.8)
#     colorbar = ax.collections[0].colorbar
#     # colorbar.fraction = 0.047*0.8
#     r = colorbar.vmax - 1
#     colorbar.set_ticks(
#         [
#             0 + r / (len(unique_labels) - 1) * (0.5 + i)
#             for i in range(1, len(unique_labels))
#         ]
#     )
#     colorbar.set_ticklabels(unique_labels[1:])

#     # ax.set(xlabel="t-SNE 1", ylabel="t-SNE 2")
#     ax.get_xaxis().set_ticks([])
#     ax.get_yaxis().set_ticks([])
#     ax.set_box_aspect(0.8)

#     ax.plot(borders[:, 0], borders[:, 1], "k.", markersize=0.5)

#     # handles = [
#     #     Line2D(
#     #         [0], [0], marker="o", color="w", markerfacecolor=v, label=k, markersize=8
#     #     )
#     #     for k, v in color_dict.items() if k!="Background"
#     # ]
#     # ax.legend(
#     #     title="color", handles=handles, bbox_to_anchor=(1.05, 1), loc="upper left"
#     # )
#     plt.savefig("./labeled_map.png")
#     plt.close()


# def feature_ridge(
#     feature: np.ndarray,
#     labels: Union[List, np.ndarray],
#     xlabel: str,
#     ylabel: str,
#     path: str = "./",
# ):
#     df = pd.DataFrame({xlabel: feature, ylabel: labels})
#     pal = sns.cubehelix_palette(10, rot=-0.25, light=0.7)
#     grid = sns.FacetGrid(df, row=ylabel, hue=ylabel, aspect=15, height=0.5, palette=pal)

#     # Draw the densities in a few steps
#     grid.map(
#         sns.kdeplot,
#         xlabel,
#         bw_adjust=0.5,
#         clip_on=False,
#         fill=True,
#         alpha=1,
#         linewidth=1.5,
#     )
#     grid.map(sns.kdeplot, xlabel, clip_on=False, color="w", lw=2, bw_adjust=0.5)

#     # passing color=None to refline() uses the hue mapping
#     grid.refline(y=0, linewidth=2, linestyle="-", color=None, clip_on=False)

#     # Define and use a simple function to label the plot in axes coordinates
#     def labelax(x, color, label):
#         ax = plt.gca()
#         ax.text(
#             0,
#             0.2,
#             label,
#             fontweight="bold",
#             color=color,
#             ha="left",
#             va="center",
#             transform=ax.transAxes,
#         )

#     grid.map(labelax, xlabel)

#     # Set the subplots to overlap
#     grid.figure.subplots_adjust(hspace=-0.25)

#     # Remove axes details that don't play well with overlap
#     grid.set_titles("")
#     grid.set(yticks=[], ylabel="")
#     grid.despine(bottom=True, left=True)

#     Path(path).mkdir(parents=True, exist_ok=True)
#     plt.savefig(path + "{}_{}_ridge.png".format(xlabel, ylabel))
