import os
import numpy as np
import scipy.io as sio
import scipy as scp
import imageio
import tqdm
import hdf5storage
import pandas as pd
import seaborn as sns
from matplotlib.lines import Line2D
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
from typing import Optional, Union, List

import DataStruct as ds
from embed import Watershed, GaussDensity
import analysis

palette = [
    (1, 0.5, 0),
    (0.5, 0.5, 0.85),
    (0, 1, 0),
    (1, 0, 0),
    (0, 0, 0.9),
    (0, 1, 1),
    (0.4, 0.4, 0.4),
    (0.5, 0.85, 0.5),
    (0.5, 0.15, 0.5),
    (0.15, 0.5, 0.5),
    (0.5, 0.5, 0.15),
    (0.9, 0.9, 0),
    (1, 0, 1),
    (0, 0.5, 1),
    (0.85, 0.5, 0.5),
    (0.5, 1, 0),
    (0.5, 0, 1),
    (1, 0, 0.5),
    (0, 0.9, 0.6),
    (0.3, 0.6, 0),
    (0, 0.3, 0.6),
    (0.6, 0.3, 0),
    (0.3, 0, 0.6),
    (0, 0.6, 0.3),
    (0.6, 0, 0.3),
]


def scatter_by_cat(
    data: np.ndarray,
    cat: np.ndarray,
    label: str,
    marker_size: int = 3,
    color = None,
    filepath: str = "./",
):
    colors = np.zeros(cat.shape)
    # for i, key in enumerate(np.unique(cat)):
    #     colors[cat==key] = np.repeat(palette[i], np.size(colors[cat==key]))
    if color == None:
        color = palette
    ax = sns.scatterplot(
        x=data[:, 0], y=data[:, 1], marker=".", hue=cat, palette= color, s=marker_size
    )
    ax.set(xlabel="t-SNE 1", ylabel="t-SNE 2")
    ax.set_box_aspect(0.8)
    ax.figure.savefig("".join([filepath, "scatter_by_", label, ".png"]))
    plt.close()
    # plt.savefig(''.join([filepath, 'scatter_by_', label, '.png']))


def scatter(
    data: Union[np.ndarray, ds.DataStruct],
    color: Optional[Union[List, np.ndarray]] = None,
    marker_size: int = 3,
    ax_label: str = "t-SNE",
    filepath: str = "./plot_folder/scatter.png",
    show: bool = False,
    **kwargs
):
    """
    Draw a 2d tSNE plot from zValues.

    input: zValues dataframe, [num of points x 2]
    output: a scatter plot
    """
    if isinstance(data, ds.DataStruct):
        x = data.embed_vals[:, 0]
        y = data.embed_vals[:, 1]
    elif isinstance(data, np.ndarray):
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
        cmap=sns.color_palette("crest", as_cmap=True),
        alpha=0.75,
        **kwargs
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
    filepath: str = "./plot_folder/watershed.png",
):
    """
    Plotting a watershed map with clusters colored

    """
    f = plt.figure()
    ax = f.add_subplot(111)
    ax.imshow(ws_map)
    ax.set_aspect("auto")
    if ws_borders is not None:
        ax.plot(ws_borders[:, 0], ws_borders[:, 1], ".k", markersize=0.05)
    plt.savefig("".join([filepath, "_watershed.png"]), dpi=200)
    plt.close()


def scatter_on_watershed(
    data: ds.DataStruct,
    watershed: GaussDensity,
    column: str,
):
    labels = data.data[column].values

    if not os.path.exists("".join([data.out_path, "points_by_cluster/"])):
        os.makedirs("".join([data.out_path, "points_by_cluster/"]))

    extent = [*watershed.hist_range[0], *watershed.hist_range[1]]

    f = plt.figure()
    ax = f.add_subplot(111)
    ax.imshow(watershed.watershed_map, zorder=1, extent=extent)
    ax.plot(
        data.embed_vals[:, 0],
        data.embed_vals[:, 1],
        ".r",
        markersize=1,
        alpha=0.1,
        zorder=2,
    )
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("auto")
    filename = "".join([data.out_path, "points_by_cluster/all.png"])
    plt.savefig(filename, dpi=200)
    plt.close()

    print("Plotting scatter on watershed for each ", column)
    for i, label in tqdm.tqdm(enumerate(np.unique(labels))):
        embed_vals = data.embed_vals[data.data[column] == label]

        f = plt.figure()
        ax = f.add_subplot(111)
        ax.imshow(watershed.watershed_map, zorder=0, extent=extent)

        ax.plot(
            embed_vals[:, 0], embed_vals[:, 1], ".r", markersize=1, alpha=0.1, zorder=1
        )
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect("auto")
        filename = "".join(
            [
                data.out_path,
                "points_by_cluster/",
                column,
                "_points_",
                str(label),
                ".png",
            ]
        )
        plt.savefig(filename, dpi=400)
        plt.close()


def density_feat(
    data: ds.DataStruct,
    watershed: Watershed,
    features: np.ndarray,
    feature_labels: List,
    key: str,
    file_path: str = "./plot_folder/",
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
    filepath: str = "./plot_folder/density.png",
    show: bool = False,
):
    f = plt.figure()
    ax = f.add_subplot(111)
    if ws_borders is not None:
        ax.plot(ws_borders[:, 0], ws_borders[:, 1], ".k", markersize=0.1)
    ax.imshow(density)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("auto")
    if filepath:
        plt.savefig(filepath, dpi=200)
    if show:
        plt.show()
    plt.close()


def density_cat(
    data: ds.DataStruct,
    column: str,
    watershed: Watershed,
    filepath: str = "./plot_folder/density_by_label.png",
    n_col: int = 4,
    show: bool = False,
):
    """
    Plot densities by a category label
    """
    labels = data.data[column].values
    n_col = min(n_col, len(np.unique(labels)))
    n_rows = int(np.ceil(len(np.unique(labels)) / n_col))
    f, ax_arr = plt.subplots(n_rows, n_col, figsize=((n_col + 1) * 4, n_rows * 4))

    # Loop over unique labels
    for i, label in enumerate(np.unique(labels)):
        embed_vals = data.embed_vals[data.data[column] == label]  # Indexing by label
        density = watershed.fit_density(
            embed_vals, new=False
        )  # Fit density on old axes
        col_i = i % n_col
        row_i = int(i/n_col)
        if n_rows == 1:
            ax_arr[col_i].imshow(density)  # scp.special.softmax(density))

            if watershed is not None:
                ax_arr[col_i].plot(
                    watershed.borders[:, 0],
                    watershed.borders[:, 1],
                    ".k",
                    markersize=0.1,
                )
            ax_arr[col_i].set_aspect(0.9)
            ax_arr[col_i].set_title(label)
            ax_arr[col_i].set_xlabel("t-SNE 1")
            ax_arr[col_i].set_ylabel("t-SNE 2")
            ax_arr[col_i].set_xticks([])
            ax_arr[col_i].set_yticks([])
        else:
            ax_arr[int(i / n_col), col_i].imshow(scp.special.softmax(density))

            if watershed is not None:
                ax_arr[row_i, col_i].plot(
                    watershed.borders[:, 0],
                    watershed.borders[:, 1],
                    ".k",
                    markersize=0.1,
                )
            ax_arr[row_i, col_i].set_aspect(0.9)
            ax_arr[row_i, col_i].set_title(label)
            ax_arr[row_i, col_i].set_xlabel("t-SNE 1")
            ax_arr[row_i, col_i].set_ylabel("t-SNE 2")
            ax_arr[row_i, col_i].set_xticks([])
            ax_arr[row_i, col_i].set_yticks([])
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
    filepath: str = "./plot_folder/density_by_label.png",
    show: bool = False,
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
        ax_arr[i, 0].set_title(label1)
        for j, label2 in enumerate(np.unique(labels2)):
            # import pdb; pdb.set_trace()
            embed_vals = data.embed_vals[
                (data.data[cat1] == label1) & (data.data[cat2] == label2)
            ]  # Indexing by label
            density = watershed.fit_density(
                embed_vals, new=False
            )  # Fit density on old axes
            if n_rows == 1:
                ax_arr[j].imshow(density)  # scp.special.softmax(density))

                if watershed is not None:
                    ax_arr[j].plot(
                        watershed.borders[:, 0],
                        watershed.borders[:, 1],
                        ".k",
                        markersize=0.1,
                    )
                ax_arr[j].set_aspect("auto")
                ax_arr[j].set_title(label1)
                ax_arr[j].set_xlabel("t-SNE 1")
                ax_arr[j].set_ylabel("t-SNE 2")
                ax_arr[j].set_xticks([])
                ax_arr[j].set_yticks([])
            else:
                ax_arr[i, j].imshow(scp.special.softmax(density))

                if watershed is not None:
                    ax_arr[i, j].plot(
                        watershed.borders[:, 0],
                        watershed.borders[:, 1],
                        ".k",
                        markersize=0.1,
                    )
                if i == 0:
                    ax_arr[0, j].set_title(label2)
                ax_arr[i, j].set_aspect("auto")
                ax_arr[i, j].set_xlabel("t-SNE 1")
                ax_arr[i, j].set_ylabel("t-SNE 2")
                ax_arr[i, j].set_xticks([])
                ax_arr[i, j].set_yticks([])
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
    # import pdb; pdb.set_trace()

    combined_labels = np.array(
        ["_".join([label1, label2]) for label1, label2 in zip(cat1_labels, cat2_labels)]
    )

    freq, combined_keys = analysis.cluster_freq_by_cat(
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
            # import pdb; pdb.set_trace()
            # cluster_labels = data_obj.data['Cluster'].values[(data_obj.data[cat1]==key1) & data_obj.data[cat2]==key2]
            # freq = analysis.cluster_freq_from_labels(cluster_labels, num_clusters)
            freq_key = "_".join([key1, key2])
            # import pdb; pdb.set_trace()
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
        # import pdb; pdb.set_trace()
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
    # import pdb; pdb.set_trace()

    combined_labels = np.array(
        ["_".join([label1, label2]) for label1, label2 in zip(cat1_labels, cat2_labels)]
    )

    freq, combined_keys = analysis.cluster_freq_by_cat(
        data_obj.data["Cluster"].values, cat=combined_labels
    )

    # cat1_freq_keys,cat2_freq_keys = map(list, zip(*[key.split(',') for key in combined_keys]))

    num_clusters = freq.shape[1]  # data_obj.data['Cluster'].max()+1
    # Unique keys for cat1 and cat2
    cat1_keys, cat2_keys = np.unique(data_obj.data[cat1].values), np.unique(
        data_obj.data[cat2].values
    )

    f, ax_arr = plt.subplots(3, 1, sharex="all", figsize=(12, 4), gridspec_kw={'height_ratios': [0.2, 2, 2]})

    for i, key1 in enumerate(cat1_keys):  # For each plot of cat1
        # videos = data_obj.meta.index[data_obj.meta['Condition'] == cat1_keys[i]].tolist()
        for j, key2 in enumerate(cat2_keys):  # For each cat2 of cat 1
            # import pdb; pdb.set_trace()
            # cluster_labels = data_obj.data['Cluster'].values[(data_obj.data[cat1]==key1) & data_obj.data[cat2]==key2]
            # freq = analysis.cluster_freq_from_labels(cluster_labels, num_clusters)
            freq_key = "_".join([key1, key2])
            # import pdb; pdb.set_trace()s
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
    ax_arr[0].set_ylim(0.59,0.6)
    ax_arr[1].spines["bottom"].set_visible(False)
    ax_arr[0].plot(range(num_clusters), freq[67,:], color=colors[0], alpha=0.1)
    ax_arr[0].get_xaxis().set_visible(False)
    ax_arr[0].spines["bottom"].set_visible(False)
    ax_arr[0].spines["right"].set_visible(False)
    ax_arr[0].spines["top"].set_visible(False)

    d = 4/4.2  # proportion of vertical to horizontal extent of the slanted line
    kwargs = dict(marker=[(-1, -d),(1,d)], markersize=12,
                linestyle="none", color='k', mec='k', mew=1, clip_on=False)
    ax_arr[0].plot([0], [0], transform=ax_arr[0].transAxes, **kwargs)
    ax_arr[1].plot([0], [1], transform=ax_arr[1].transAxes, **kwargs)
    markers = ["o", "v", "s"]
    # for j in range(len(conditions),len(conditions)+2): # For mean and variance plots
    j = 1
    for i, key1 in enumerate(cat1_keys):  # for each condition
        # videos = data.meta.index[data.meta['Condition'] == conditions[j]].tolist()
        key_bool = [True if key.startswith(key1) else False for key in combined_keys]
        # import pdb; pdb.set_trace()
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


def skeleton_vid3D_cat(
    data: ds.DataStruct,
    column: str,
    labels: Optional[List] = None,
    n_skeletons: int = 9,
    filepath: str = "./plot_folder",
):

    col_vals = data.data[column].values
    index = np.arange(len(col_vals))
    if labels is None:
        labels = np.unique(col_vals)

    for label in tqdm.tqdm(labels):
        label_idx = index[col_vals == label]
        if len(label_idx) == 0:
            continue
        else:
            num_points = min(len(label_idx), n_skeletons)
            permuted_points = data.frame[
                np.random.permutation(label_idx)
            ]  # b/c moving frames filter
            sampled_points = []
            for i in range(len(permuted_points)):
                # import pdb; pdb.set_trace()
                if i == 0:
                    # print("first idx")
                    sampled_points = np.array([permuted_points[i]])
                    # print(sampled_points)
                    continue
                elif len(sampled_points) == num_points:  # sampled enough points
                    break
                elif any(
                    np.abs(permuted_points[i] - sampled_points) < 200
                ):  # point is not far enough from previous points
                    continue
                else:
                    sampled_points = np.append(sampled_points, permuted_points[i])

            print(sampled_points)
            # import pdb; pdb.set_trace()
            skeleton_vid3D_single(
                data,
                # label=label,
                frames=sampled_points,
                N_FRAMES=100,
                VID_NAME="".join([column, "_", str(label), ".mp4"]),
                SAVE_ROOT="".join([filepath, "/skeleton_vids/"]),
            )


def skeleton_vid3D_expanded(
    data: Union[ds.DataStruct, np.ndarray],
    label: str,
    connectivity: Optional[ds.Connectivity] = None,
    frames: List = [3000, 100000, 5000000],
    N_FRAMES: int = 150,
    fps: int = 90,
    dpi: int = 100,
    VID_NAME: str = "0.mp4",
    SAVE_ROOT: str = "./test/skeleton_vids/",
):

    if isinstance(data, ds.DataStruct):
        preds = data.pose
        connectivity = data.connectivity
    else:
        preds = data

    START_FRAME = np.array(frames) - int(N_FRAMES / 2) + 1
    COLOR = connectivity.colors * len(frames)
    links = connectivity.links
    links_expand = links
    # total_frames = N_FRAMES*len(frames)#max(np.shape(f[list(f.keys())[0]]))

    ## Expanding connectivity for each frame to be visualized
    num_joints = max(max(links)) + 1
    for i in range(len(frames) - 1):
        next_con = [
            (x + (i + 1) * num_joints, y + (i + 1) * num_joints) for x, y in links
        ]
        links_expand = links_expand + next_con

    save_path = os.path.join(SAVE_ROOT)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # get dannce predictions
    pose_3d = np.empty((0, num_joints, 3))
    for start in START_FRAME:
        pose_3d = np.append(pose_3d, preds[start : start + N_FRAMES, :, :], axis=0)

    # compute 3d grid limits
    offset = 50
    x_lim1, x_lim2 = (
        np.min(pose_3d[:, :, 0]) - offset,
        np.max(pose_3d[:, :, 0]) + offset,
    )
    y_lim1, y_lim2 = (
        np.min(pose_3d[:, :, 1]) - offset,
        np.max(pose_3d[:, :, 1]) + offset,
    )
    z_lim1, z_lim2 = (
        np.minimum(0, np.min(pose_3d[:, :, 2])),
        np.max(pose_3d[:, :, 2]) + 10,
    )

    # set up video writer
    metadata = dict(title="dannce_visualization", artist="Matplotlib")
    writer = FFMpegWriter(fps=fps)  # , metadata=metadata)

    extent = [*data.ws.hist_range[0], *data.ws.hist_range[1]]
    embed_vals = data.embed_vals[data.data["Cluster"] == label]
    colors = [
        "k",
        "tab:blue",
        "tab:green",
        "tab:orange",
        "tab:purple",
        "tab:brown",
        "tab:pink",
        "tab:gray",
        "tab:olive",
        "tab:cyan",
        "#dc0ab4",
        "#00b7c7",
    ]

    # Setup figure
    fig = plt.figure(figsize=(20, 10))
    gs = fig.add_gridspec(1, 2)
    ax_3d = fig.add_subplot(gs[0, 1], projection="3d")
    ax_dens = fig.add_subplot(gs[0, 0])

    cond_uniq = data.data["Condition"].unique()
    frames_meta = data.data[data.data["frame"].isin(frames)]["Condition"].values

    ax_dens.imshow(data.ws.watershed_map, zorder=0, extent=extent)
    ax_dens.plot(
        embed_vals[:, 0], embed_vals[:, 1], ".r", markersize=1, alpha=0.1, zorder=1
    )
    ax_dens.set_aspect("auto")
    ax_dens.set_xticks([])
    ax_dens.set_yticks([])

    with writer.saving(fig, os.path.join(save_path, "vis_" + VID_NAME), dpi=dpi):

        for curr_frame in tqdm.tqdm(range(N_FRAMES)):
            # grab frames
            curr_frames = curr_frame + np.arange(len(frames)) * N_FRAMES
            kpts_3d = np.reshape(
                pose_3d[curr_frames, :, :], (len(frames) * num_joints, 3)
            )

            # plot 3d moving skeletons
            for i in range(len(np.unique(frames_meta))):
                temp_idx = curr_frames[frames_meta == np.unique(frames_meta)[i]]
                temp_kpts = np.reshape(
                    [pose_3d[temp_idx, :, :]], (len(temp_idx) * num_joints, 3)
                )
                ax_3d.scatter(
                    temp_kpts[:, 0],
                    temp_kpts[:, 1],
                    temp_kpts[:, 2],
                    marker=".",
                    color=colors[i],
                    linewidths=0.5,
                    label=cond_uniq[i],
                )
            ax_3d.legend()
            for color, (index_from, index_to) in zip(COLOR, links_expand):
                xs, ys, zs = [
                    np.array([kpts_3d[index_from, j], kpts_3d[index_to, j]])
                    for j in range(3)
                ]
                ax_3d.plot3D(xs, ys, zs, c=color, lw=2)

            ax_3d.set_xlim(x_lim1, x_lim2)
            ax_3d.set_ylim(y_lim1, y_lim2)
            ax_3d.set_zlim(0, 150)
            ax_3d.set_xlabel("x")
            ax_3d.set_ylabel("y")
            # ax_3d.set_xticks([])
            # ax_3d.set_yticks([])
            # ax_3d.set_zticks([])
            # ax_3d.set_title("3D Tracking")
            # ax_3d.set_aspect('equal')
            ax_3d.set_box_aspect([1, 1, 0.4])

            # grab frame and write to vid
            writer.grab_frame()
            fig.tight_layout()
            ax_3d.clear()

    plt.close()
    return 0


def skeleton_vid3D(
    data: Union[ds.DataStruct, np.ndarray],
    connectivity: Optional[ds.Connectivity] = None,
    frames: List = [3000, 100000, 500000],
    N_FRAMES: int = 300,
    fps: int = 90,
    dpi: int = 200,
    VID_NAME: str = "0.mp4",
    SAVE_ROOT: str = "./test/skeleton_vids/",
):

    if isinstance(data, ds.DataStruct):
        preds = data.pose
        connectivity = data.connectivity
    else:
        preds = data

    if connectivity is None:
        skeleton_name = "mouse" + str(preds.shape[1])
        connectivity = Connectivity().load(
            "../../CAPTURE_data/skeletons.py", skeleton_name=skeleton_name
        )

    START_FRAME = np.array(frames) - int(N_FRAMES / 2) + 1
    COLOR = connectivity.colors * len(frames)
    links = connectivity.links
    links_expand = links
    # total_frames = N_FRAMES*len(frames)#max(np.shape(f[list(f.keys())[0]]))

    ## Expanding connectivity for each frame to be visualized
    num_joints = max(max(links)) + 1
    for i in range(len(frames) - 1):
        next_con = [
            (x + (i + 1) * num_joints, y + (i + 1) * num_joints) for x, y in links
        ]
        links_expand = links_expand + next_con

    save_path = os.path.join(SAVE_ROOT)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # get dannce predictions
    pose_3d = np.empty((0, num_joints, 3))
    for start in START_FRAME:
        pose_3d = np.append(pose_3d, preds[start : start + N_FRAMES, :, :], axis=0)

    # compute 3d grid limits
    offset = 50
    x_lim1, x_lim2 = (
        np.min(pose_3d[:, :, 0]) - offset,
        np.max(pose_3d[:, :, 0]) + offset,
    )
    y_lim1, y_lim2 = (
        np.min(pose_3d[:, :, 1]) - offset,
        np.max(pose_3d[:, :, 1]) + offset,
    )
    z_lim1, z_lim2 = (
        np.minimum(0, np.min(pose_3d[:, :, 2])),
        np.max(pose_3d[:, :, 2]) + 10,
    )

    # set up video writer
    # metadata = dict(title='dannce_visualization', artist='Matplotlib')
    writer = FFMpegWriter(fps=fps)  # , metadata=metadata)

    # Setup figure
    fig = plt.figure(figsize=(12, 12))
    ax_3d = fig.add_subplot(1, 1, 1, projection="3d")
    with writer.saving(fig, os.path.join(save_path, "vis_" + VID_NAME), dpi=dpi):
        for curr_frame in tqdm.tqdm(range(N_FRAMES)):
            # grab frames
            curr_frames = curr_frame + np.arange(len(frames)) * N_FRAMES
            kpts_3d = np.reshape(
                pose_3d[curr_frames, :, :], (len(frames) * num_joints, 3)
            )

            # plot 3d moving skeletons
            ax_3d.scatter(
                kpts_3d[:, 0],
                kpts_3d[:, 1],
                kpts_3d[:, 2],
                marker=".",
                color="black",
                linewidths=0.5,
            )
            for color, (index_from, index_to) in zip(COLOR, links_expand):
                xs, ys, zs = [
                    np.array([kpts_3d[index_from, j], kpts_3d[index_to, j]])
                    for j in range(3)
                ]
                ax_3d.plot3D(xs, ys, zs, c=color, lw=2)

            ax_3d.set_xlim(x_lim1, x_lim2)
            ax_3d.set_ylim(y_lim1, y_lim2)
            ax_3d.set_zlim(0, 150)
            ax_3d.set_title("3D Tracking")
            ax_3d.set_xlabel("x")
            ax_3d.set_ylabel("y")
            # ax_3d.set_aspect('equal')
            ax_3d.set_box_aspect([1, 1, 0.4])

            # grab frame and write to vid
            writer.grab_frame()
            ax_3d.clear()

    plt.close()
    return 0

def skeleton_vid3D_single(
    data: Union[ds.DataStruct, np.ndarray],
    connectivity: Optional[ds.Connectivity] = None,
    frames: List = [3000, 100000, 500000],
    N_FRAMES: int = 300,
    fps: int = 90,
    dpi: int = 200,
    VID_NAME: str = "0.mp4",
    SAVE_ROOT: str = "./test/skeleton_vids/",
):

    if isinstance(data, ds.DataStruct):
        preds = data.pose
        connectivity = data.connectivity
    else:
        preds = data

    if connectivity is None:
        skeleton_name = "mouse" + str(preds.shape[1])
        connectivity = Connectivity().load(
            "../../CAPTURE_data/skeletons.py", skeleton_name=skeleton_name
        )

    START_FRAME = np.array(frames) - int(N_FRAMES / 2) + 1
    COLOR = connectivity.colors * len(frames)
    links = connectivity.links
    links_expand = links
    # total_frames = N_FRAMES*len(frames)#max(np.shape(f[list(f.keys())[0]]))

    ## Expanding connectivity for each frame to be visualized
    num_joints = max(max(links)) + 1
    for i in range(len(frames) - 1):
        next_con = [
            (x + (i + 1) * num_joints, y + (i + 1) * num_joints) for x, y in links
        ]
        links_expand = links_expand + next_con

    save_path = os.path.join(SAVE_ROOT)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # get dannce predictions
    pose_3d = np.empty((0, num_joints, 3))
    for start in START_FRAME:
        pose_3d = np.append(pose_3d, preds[start : start + N_FRAMES, :, :], axis=0)

    # compute 3d grid limits
    offset = 50
    x_lim1, x_lim2 = (
        np.min(pose_3d[:, :, 0]) - offset,
        np.max(pose_3d[:, :, 0]) + offset,
    )
    y_lim1, y_lim2 = (
        np.min(pose_3d[:, :, 1]) - offset,
        np.max(pose_3d[:, :, 1]) + offset,
    )
    z_lim1, z_lim2 = (
        np.minimum(0, np.min(pose_3d[:, :, 2])),
        np.max(pose_3d[:, :, 2]) + 10,
    )

    # set up video writer
    # metadata = dict(title='dannce_visualization', artist='Matplotlib')
    writer = FFMpegWriter(fps=fps)  # , metadata=metadata)

    # Setup figure
    fig = plt.figure(figsize=(12, 12))
    ax_3d = fig.add_subplot(1, 1, 1, projection="3d")
    with writer.saving(fig, os.path.join(save_path, "vis_" + VID_NAME), dpi=dpi):
        for curr_frame in tqdm.tqdm(range(N_FRAMES)):
            # grab frames
            curr_frames = curr_frame + np.arange(len(frames)) * N_FRAMES
            kpts_3d = np.reshape(
                pose_3d[curr_frames, :, :], (len(frames) * num_joints, 3)
            )

            # plot 3d moving skeletons
            ax_3d.scatter(
                kpts_3d[:, 0],
                kpts_3d[:, 1],
                kpts_3d[:, 2],
                marker=".",
                s=200,
                color="black",
                linewidths=0.5,
            )
            for color, (index_from, index_to) in zip(COLOR, links_expand):
                xs, ys, zs = [
                    np.array([kpts_3d[index_from, j], kpts_3d[index_to, j]])
                    for j in range(3)
                ]
                ax_3d.plot3D(xs, ys, zs, c=color, lw=3)

            # ax_3d.set_xlim(x_lim1, x_lim2)
            # ax_3d.set_ylim(y_lim1, y_lim2)
            # ax_3d.set_zlim(0, 150)
            # ax_3d.set_title("3D Tracking")
            # ax_3d.set_xlabel("x")
            # ax_3d.set_ylabel("y")
            # ax_3d.set_aspect('equal')
            ax_3d.set_box_aspect([1, 1, 1])
            ax_3d.axis('off')
            # ax_3d.grid(b=None)

            # grab frame and write to vid
            writer.grab_frame()
            ax_3d.clear()

    plt.close()
    return 0


def feature_hist(feature, label, filepath, range=None):
    plt.hist(feature, bins=1000, range=range, density=True)
    plt.xlabel(label)
    plt.ylabel("Histogram Density")
    if filepath:
        plt.savefig("".join([filepath, label, "_hist.png"]))
    plt.close()
    return


def skeleton_vid3D_features(
    pose,
    feature,
    connectivity: Optional[ds.Connectivity] = None,
    frames: List = [3000],
    N_FRAMES: int = 150,
    fps: int = 90,
    dpi: int = 200,
    VID_NAME: str = "0.mp4",
    SAVE_ROOT: str = "./test/skeleton_vids/",
):

    START_FRAME = np.array(frames) - int(N_FRAMES / 2) + 1
    COLOR = connectivity.colors * len(frames)
    links = connectivity.links
    links_expand = links

    ## Expanding connectivity for each frame to be visualized
    num_joints = max(max(links)) + 1
    for i in range(len(frames) - 1):
        next_con = [
            (x + (i + 1) * num_joints, y + (i + 1) * num_joints) for x, y in links
        ]
        links_expand = links_expand + next_con

    save_path = os.path.join(SAVE_ROOT)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # get dannce predictions
    pose_3d = np.empty((0, num_joints, 3))
    for start in START_FRAME:
        pose_3d = np.append(pose_3d, pose[start : start + N_FRAMES, :, :], axis=0)
        feature = feature[start : start + N_FRAMES]

    # compute 3d grid limits
    offset = 10
    x_lim1, x_lim2 = (
        np.min(pose_3d[:, :, 0]) - offset,
        np.max(pose_3d[:, :, 0]) + offset,
    )
    y_lim1, y_lim2 = (
        np.min(pose_3d[:, :, 1]) - offset,
        np.max(pose_3d[:, :, 1]) + offset,
    )
    z_lim1, z_lim2 = (
        np.minimum(0, np.min(pose_3d[:, :, 2])),
        np.max(pose_3d[:, :, 2]) + 10,
    )

    # set up video writer
    writer = FFMpegWriter(fps=int(fps / 4))

    # Setup figure
    fig = plt.figure(figsize=(20, 10))
    gs = fig.add_gridspec(1, 2)
    ax_3d = fig.add_subplot(gs[0, 1], projection="3d")
    ax_trace = fig.add_subplot(gs[0, 0])

    with writer.saving(fig, os.path.join(save_path, "vis_feat_" + VID_NAME), dpi=dpi):

        for curr_frame in tqdm.tqdm(range(N_FRAMES)):
            # grab frames
            curr_frames = curr_frame + np.arange(len(frames)) * N_FRAMES

            ax_trace.plot(
                np.arange(curr_frames + 1),
                feature[: curr_frames[0] + 1],
                linestyle="-",
                linewidth=1,
            )
            ax_trace.plot(
                curr_frames, feature[curr_frames], marker=".", markersize=20, color="k"
            )

            kpts_3d = np.reshape(
                pose_3d[curr_frames, :, :], (len(frames) * num_joints, 3)
            )

            # plot 3d moving skeletons
            ax_3d.scatter(
                kpts_3d[:, 0],
                kpts_3d[:, 1],
                kpts_3d[:, 2],
                marker=".",
                color="black",
                linewidths=0.5,
            )
            for color, (index_from, index_to) in zip(COLOR, links_expand):
                xs, ys, zs = [
                    np.array([kpts_3d[index_from, j], kpts_3d[index_to, j]])
                    for j in range(3)
                ]
                ax_3d.plot3D(xs, ys, zs, c=color, lw=2)

            ax_3d.set_xlim(x_lim1, x_lim2)
            ax_3d.set_ylim(y_lim1, y_lim2)
            ax_3d.set_zlim(0, 150)
            ax_3d.set_xlabel("x")
            ax_3d.set_ylabel("y")
            # ax_3d.set_xticks([])
            # ax_3d.set_yticks([])
            # ax_3d.set_zticks([])
            # ax_3d.set_title("3D Tracking")
            # ax_3d.set_aspect('equal')
            ax_3d.set_box_aspect([1, 1, 0.4])

            # grab frame and write to vid
            writer.grab_frame()
            fig.tight_layout()
            ax_3d.clear()

    plt.close()
    return 0


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

        skeleton_vid3D(
            data_obj.pose,
            data_obj.connectivity,
            frames=[sorted_idx[-1] * 10],
            N_FRAMES=100,
            dpi=100,
            VID_NAME="highest_" + heur_key + "_score.mp4",
            SAVE_ROOT=filepath,
        )


def labeled_watershed(watershed, borders, labels_path):

    labels = pd.read_csv(labels_path)

    behavior_cats = set(labels["Label"])

    color_dict = {
        "Background": np.array([0, 0, 0]),
        "Crouched Idle": np.array([241, 194, 50]),
        "Face Groom": np.array([106, 168, 79]),
        "Body Groom": np.array([106, 168, 79]),
        "High Rear": np.array([69, 129, 142]),
        "Low Rear": np.array([106, 168, 79]),
        "Investigate": np.array([106, 168, 79]),
        "Walk/Rear": np.array([106, 168, 79]),
        "Walk": np.array([103, 78, 167]),
    }

    colors = [
        "#FFFFFF",
        "tab:gray",
        "tab:green",
        "tab:blue",
        "tab:orange",
        "tab:red",
        "tab:purple",
        "tab:brown",
        "tab:pink",
        "tab:olive",
        "tab:cyan",
        "#dc0ab4",
        "#00b7c7",
    ]

    unique_labels = labels["Label"].unique()

    labeled_map = np.zeros(watershed.shape)
    for i, label in enumerate(unique_labels):
        # import pdb; pdb.set_trace()
        labeled_map[np.isin(watershed,labels["Cluster"][labels["Label"] == label].values)] = i



    # labeled_map = np.zeros((watershed.shape[0], watershed.shape[1], 3))
    # for i in range(watershed.shape[0]):
    #     for j in range(watershed.shape[1]):
    #         # import pdb; pdb.set_trace()
    #         # try:
    #         label = labels["Label"].loc[labels["Cluster"] == watershed[i, j]].values[0]
    #         # except:
    #         #     import pdb; pdb.set_trace()
    #         labeled_map[i, j, :] = np.array(color_dict[label])

    # fig, ax = plt.subplots()
    sns.set(rc={'figure.figsize':(12,10)})
    cmap =[(1,1,1)] + sns.color_palette("Pastel2", len(unique_labels)-1) 
    # import pdb; pdb.set_trace()
    ax = sns.heatmap(labeled_map, cmap=cmap)
    plt.colorbar(ax=ax, fraction = 0.047*0.8)
    colorbar = ax.collections[0].colorbar 
    # colorbar.fraction = 0.047*0.8
    r = colorbar.vmax - 1
    colorbar.set_ticks([0 + r / (len(unique_labels)-1) * (0.5 + i) for i in range(1,len(unique_labels))])
    colorbar.set_ticklabels(unique_labels[1:])
    
    # ax.set(xlabel="t-SNE 1", ylabel="t-SNE 2")
    ax.get_xaxis().set_ticks([])
    ax.get_yaxis().set_ticks([])
    ax.set_box_aspect(0.8)

    ax.plot(borders[:, 0], borders[:, 1], "k.", markersize=0.5)

    # handles = [
    #     Line2D(
    #         [0], [0], marker="o", color="w", markerfacecolor=v, label=k, markersize=8
    #     )
    #     for k, v in color_dict.items() if k!="Background"
    # ]
    # ax.legend(
    #     title="color", handles=handles, bbox_to_anchor=(1.05, 1), loc="upper left"
    # )
    plt.savefig("./labeled_map.png")
    plt.close()
