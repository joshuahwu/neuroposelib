import numpy as np
import tqdm

import pandas as pd

# import seaborn as sns
from matplotlib.lines import Line2D
import matplotlib
from pathlib import Path

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from typing import Optional, Union, List, Dict, Tuple, Any, Sequence
from scipy.special import softmax
from sklearn.preprocessing import MinMaxScaler

from neuroposelib import DataStruct as ds
from neuroposelib.embed import Watershed, GaussDensity
from neuroposelib.analysis import hist_cluster_by_cat
from neuroposelib.visualization.constants import (
    PALETTE,
    EPS,
    DEFAULT_VIRIDIS,
    CUSTOM_CMAPS,
)
import numpy.typing as npt
import copy
from matplotlib.patches import Patch
from skimage import measure


def _hex_to_rgb(hex_color: str) -> Optional[npt.NDArray[np.float64]]:
    """
    Convert a hex color string to an RGB numpy array scaled 0..1.

    Parameters
    ----------
    hex_color : str
        Hex color string, with or without leading '#', e.g. '#ff00aa' or 'ff00aa'.

    Returns
    -------
    rgb : ndarray(float64), shape (3,)
        RGB values in range [0,1], or None if input invalid.
    """
    hex_color = hex_color.lstrip("#")
    if len(hex_color) != 6:
        return None
    try:
        r = int(hex_color[0:2], 16)
        g = int(hex_color[2:4], 16)
        b = int(hex_color[4:6], 16)
        return np.array([r / 255.0, g / 255.0, b / 255.0], dtype=np.float64)
    except ValueError:
        return None


def scatter(
    data: npt.NDArray[Any],
    color: Optional[npt.ArrayLike] = None,
    marker_size: int = 3,
    ax_label: str = "t-SNE",
    filepath: str = "./results/scatter.png",
    show: bool = False,
    **kwargs: Any,
) -> None:
    """
    Draw a 2D scatter plot from 2D embedding values.

    Parameters
    ----------
    data : ndarray, shape (N, 2) or (2, N)
        2D coordinates to plot. Either Nx2 (rows = points) or 2xN (rows = dims).
    color : array-like or None
        Color/colormap information passed to `plt.scatter`. Can be numeric array or
        colormap string depending on usage.
    marker_size : int, default 3
        Scatter marker size.
    ax_label : str, default "t-SNE"
        Label prefix for axes (e.g., 't-SNE' results in 't-SNE 1' and 't-SNE 2').
    filepath : str, default "./results/scatter.png"
        Path to save the figure. If empty string, the figure is not saved.
    show : bool, default False
        If True, call `plt.show()` before closing.
    **kwargs : dict
        Additional keyword args forwarded to `plt.scatter`.

    Returns
    -------
    None
    """
    data = np.asarray(data)
    # Support either (N,2) or (2,N)
    if data.ndim != 2:
        raise ValueError("`data` must be 2D array with shape (N,2) or (2,N).")
    if data.shape[1] == 2:
        x = data[:, 0]
        y = data[:, 1]
    elif data.shape[0] == 2:
        x = data[0, :]
        y = data[1, :]
    else:
        raise ValueError(
            "`data` must have second dimension size 2 (Nx2) or first dimension 2 (2xN)."
        )

    f = plt.figure()
    plt.scatter(
        x,
        y,
        marker=".",
        s=marker_size,
        linewidths=0,
        c=color,
        alpha=0.75,
        **kwargs,
    )
    plt.xlabel(ax_label + " 1")
    plt.ylabel(ax_label + " 2")
    if color is not None:
        plt.colorbar()
    if filepath:
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(filepath, dpi=200)

    if show:
        plt.show()
    plt.close()


def watershed(
    ws_map: npt.NDArray[Any],
    ws_borders: Optional[Dict[int, npt.NDArray[Any]]] = None,
    cmap: Optional[str] = None,
    filepath: str = "./results/watershed.png",
) -> None:
    """
    Plot a watershed map with optional borders and cluster labels.

    Parameters
    ----------
    ws_map : ndarray, shape (H, W)
        Integer labelled watershed map (cluster labels per pixel).
    ws_borders : dict(int -> ndarray), optional
        Mapping from cluster id to border coordinates array (Ncoords x 2).
    cmap : str, optional
        Colormap name for imshow.
    filepath : str, default "./results/watershed.png"
        File path to save the figure.

    Returns
    -------
    None
    """
    if cmap is None:
        cmap = DEFAULT_VIRIDIS
    f = plt.figure()
    ax = f.add_subplot(111)
    ax.imshow(ws_map, cmap=cmap)
    ax.set_aspect(0.9)
    if ws_borders is not None:
        for k, v in ws_borders.items():
            ax.plot(v[:, 0], v[:, 1], "k", markersize=0, lw=0.25)
            cluster_loc = np.where(ws_map == k)
            cluster_loc = [np.mean(inds) for inds in cluster_loc]
            ax.text(
                cluster_loc[1],
                cluster_loc[0],
                str(k),
                horizontalalignment="center",
                verticalalignment="center",
            )

    ax.axis("off")
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(filepath, dpi=200)
    plt.close()


def scatter_on_watershed(
    data: ds.DataStruct,
    watershed: GaussDensity,
    column: str,
    path: str = "./results/",
) -> None:
    """
    Overlay scatter points (embed_vals) on top of a watershed map and save per-category images.

    Parameters
    ----------
    data : DataStruct
        DataStruct containing `.embed_vals` (n_frames x 2) and `.data` with categorical columns.
    watershed : GaussDensity
        Watershed/GaussDensity object that provides `watershed_map` and `hist_range`.
    column : str
        Column name in `data.data` to split plots by.
    path : str, default "./results/"
        Output directory prefix.

    Returns
    -------
    None
    """
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
    features: npt.NDArray[Any],
    feature_labels: List[str],
    key: str,
    file_path: str = "./results/",
) -> None:
    """
    Build and plot a feature heatmap mapped to watershed bins.

    Parameters
    ----------
    data : DataStruct
        DataStruct providing `.embed_vals`.
    watershed : Watershed
        Watershed object used for binning.
    features : ndarray, shape (n_frames, n_features)
        Feature matrix aligned to `data`.
    feature_labels : list of str
        Labels for columns in `features`.
    key : str
        Feature name to map.
    file_path : str, default "./results/"
        Output directory.

    Returns
    -------
    None
    """
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
    density: npt.NDArray[Any],
    ws_borders: Optional[Dict[int, npt.NDArray[Any]]] = None,
    filepath: str = "./results/density.png",
    cmap: Optional[str] = None,
    show: bool = False,
    vmax: Optional[float] = None,
    return_fig: bool = False,
) -> Optional[plt.Figure]:
    """
    Render a 2D density map with optional watershed borders.

    Parameters
    ----------
    density : ndarray, shape (H, W)
        Density image to show.
    ws_borders : dict(int -> ndarray), optional
        Borders mapping for overlay.
    filepath : str, default "./results/density.png"
        Save location.
    cmap : str, optional
        Colormap name.
    show : bool, default False
        Show figure interactively before closing.
    vmax : float, optional
        Max for color scale.
    return_fig : bool, default False
        If True, return the matplotlib Figure object instead of closing it.

    Returns
    -------
    fig : matplotlib.figure.Figure or None
        Figure object when `return_fig` is True, otherwise None.
    """
    vmin = 0.99 * 15 / density.shape[0] ** 2
    density_copy = np.copy(density)
    density_copy[density_copy < vmin] = -np.inf
    f = plt.figure()
    ax = f.add_subplot(111)
    if cmap is None:
        cmap = DEFAULT_VIRIDIS
    if ws_borders is not None:
        for k, v in ws_borders.items():
            ax.plot(v[:, 0], v[:, 1], "k", markersize=0, lw=0.25)

    ax.imshow(density_copy, vmin=vmin, vmax=vmax, cmap=cmap)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect(0.9)
    ax.axis("off")
    if filepath:
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(filepath, dpi=200)
    if show:
        plt.show()

    if return_fig:
        return f
    else:
        plt.close()
        return None


def _mask_density(
    density: npt.NDArray[Any], watershed_map: npt.NDArray[Any], eps: float = EPS * 1.01
) -> npt.NDArray[Any]:
    """
    Apply mask to density: keep values inside watershed (>0) and clip outside to -inf.

    Parameters
    ----------
    density : ndarray, shape (H, W)
    watershed_map : ndarray, shape (H, W)
    eps : float
        Minimal value to enforce inside mask.

    Returns
    -------
    density_masked : ndarray, shape (H, W)
    """
    mask = watershed_map > 0
    density[mask] = np.maximum(density[mask], eps)
    density[~mask] = -np.inf
    return density


def density_cat(
    data: ds.DataStruct,
    column: str,
    watershed: Watershed,
    filepath: str = "./results/density_by_label.png",
    show: bool = False,
    vmax: Optional[float] = None,
) -> None:
    """
    Plot density maps for each unique label in `column`.

    Parameters
    ----------
    data : DataStruct
    column : str
        Column in data.data to group by.
    watershed : Watershed
    filepath : str
        Output file path (one image with grid).
    show : bool, default False
    vmax : float, optional
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
        )

        if watershed is not None:
            for k, v in watershed.borders.items():
                ax.plot(v[:, 0], v[:, 1], "k", markersize=0, lw=0.25)
        ax.set_aspect(0.9)
        ax.set_title(label)

    for ax in ax_arr.reshape(-1):
        ax.set_xticks([])
        ax.set_yticks([])
        ax.axis("off")

    f.tight_layout()
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(filepath, dpi=200)
    if show:
        plt.show()
    plt.close()
    return None


def density_grid(
    data: ds.DataStruct,
    cat1: str,
    cat2: str,
    watershed: Watershed,
    col_cmaps: Optional[List[str]] = None,
    filepath: str = "./results/density_by_label.png",
    show: bool = False,
    vmax: float = 3.5,
    return_fig: bool = False,
) -> Optional[plt.Figure]:
    """
    Create a grid of density maps arranged by two categorical variables.

    Parameters
    ----------
    data : DataStruct
    cat1 : str
    cat2 : str
    watershed : Watershed
    col_cmaps : list of str, optional
        Per-column colormaps to use.
    filepath : str
    show : bool
    vmax : float
    return_fig : bool
        If True, return the Figure object.

    Returns
    -------
    fig : matplotlib.figure.Figure or None
    """
    labels1, labels2 = data.data[cat1].values, data.data[cat2].values
    n_col = len(np.unique(labels2))
    n_rows = len(np.unique(labels1))
    f, ax_arr = plt.subplots(n_rows, n_col, figsize=((n_col + 1) * 4, n_rows * 4))

    # Loop over unique labels
    for i, label1 in enumerate(np.unique(labels1)):
        for j, label2 in enumerate(np.unique(labels2)):
            cmap = DEFAULT_VIRIDIS if not col_cmaps else col_cmaps[j]
            # Indexing latent embedding by label
            embed_vals = data.embed_vals[
                (data.data[cat1] == label1) & (data.data[cat2] == label2)
            ]
            if len(embed_vals) > 0:
                density = watershed.fit_density(embed_vals, new=False)
                idx = i * len(np.unique(labels2)) + j
                ax_arr.ravel()[idx].imshow(
                    _mask_density(density, watershed.watershed_map, EPS * 1.01),
                    vmin=EPS,
                    vmax=vmax,
                    cmap=cmap,
                )

                if watershed is not None:
                    for k, v in watershed.borders.items():
                        ax_arr.ravel()[idx].plot(
                            v[:, 0], v[:, 1], "k", markersize=0, lw=0.25
                        )
                ax_arr.ravel()[idx].set_aspect(0.9)
                ax_arr.ravel()[idx].set_xticks([])
                ax_arr.ravel()[idx].set_yticks([])
                for spine in ax_arr.ravel()[idx].spines.values():
                    spine.set_visible(False)

                if j == 0:
                    ax_arr.ravel()[idx].set_ylabel(label1)

                if i == 0:
                    ax_arr.ravel()[idx].set_title(label2)

    if return_fig:
        if show:
            plt.show()
        return f
    else:
        f.tight_layout()
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(filepath, dpi=200)
        if show:
            plt.show()
        plt.close()
        return None


def cluster_freq(
    data_obj: ds.DataStruct,
    cat1: str,
    cat2: str,
    filepath: str = "./",
    show: bool = False,
) -> None:
    """
    Plot cluster frequency by two categorical variables.

    Parameters
    ----------
    data_obj : DataStruct
    cat1 : str
        Primary categorical variable (one subplot per unique value).
    cat2 : str
        Secondary categorical variable (one line per unique value per subplot).
    filepath : str
        Directory to save results.
    show : bool
        If True, show plots interactively.

    Returns
    -------
    None
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

    num_clusters = freq.shape[1]
    cat1_keys, cat2_keys = np.unique(data_obj.data[cat1].values), np.unique(
        data_obj.data[cat2].values
    )

    f, ax_arr = plt.subplots(len(cat1_keys) + 1, 1, sharex="all", figsize=(20, 10))

    for i, key1 in enumerate(cat1_keys):
        for j, key2 in enumerate(cat2_keys):
            freq_key = "_".join([key1, key2])
            ax_arr[i].plot(
                range(num_clusters),
                np.squeeze(freq[combined_keys == freq_key, :]),
                label=key2,
            )

        ax_arr[i].set_title(cat1_keys[i], pad=-14)
        ax_arr[i].spines["top"].set_visible(False)
        ax_arr[i].get_xaxis().set_visible(False)
        ax_arr[i].spines["right"].set_visible(False)
        ax_arr[i].spines["bottom"].set_visible(False)
    ax_arr[1].set_ylabel("% Time Spent in Cluster")
    ax_arr[0].legend(loc="upper right", ncol=6)

    markers = ["o", "v", "s"]
    j = len(cat1_keys)
    for i, key1 in enumerate(cat1_keys):
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
    ax_arr[j].spines["top"].set_visible(False)
    ax_arr[j].spines["right"].set_visible(False)

    ax_arr[j].legend(loc="upper right", ncol=3)
    ax_arr[j].set_ylabel("Mean")
    ax_arr[j].set_xlabel("Cluster")
    ax_arr[j].set_xlim([-0.25, freq.shape[1] + 0.25])
    f.tight_layout()
    plt.subplots_adjust(hspace=0.05)
    Path(filepath).mkdir(parents=True, exist_ok=True)
    plt.savefig("".join([filepath, "mean_sd_cluster_freq.png"]), dpi=200)
    if show:
        plt.show()

    plt.close()
    return None


def cluster_freq_cond(
    data_obj: ds.DataStruct,
    cat1: str,
    cat2: str,
    filepath: str = "./",
    show: bool = False,
) -> None:
    """
    Variant of cluster_freq with fixed layout for condition plots.

    Parameters
    ----------
    data_obj : DataStruct
    cat1 : str
    cat2 : str
    filepath : str
    show : bool

    Returns
    -------
    None
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
    cat1_labels = data_obj.data[cat1].astype(str).values.tolist()
    cat2_labels = data_obj.data[cat2].astype(str).values.tolist()

    combined_labels = np.array(
        ["_".join([label1, label2]) for label1, label2 in zip(cat1_labels, cat2_labels)]
    )

    freq, combined_keys = hist_cluster_by_cat(
        data_obj.data["Cluster"].values, cat=combined_labels
    )

    num_clusters = freq.shape[1]
    cat1_keys, cat2_keys = np.unique(data_obj.data[cat1].values), np.unique(
        data_obj.data[cat2].values
    )

    f, ax_arr = plt.subplots(
        3, 1, sharex="all", figsize=(12, 4), gridspec_kw={"height_ratios": [0.2, 2, 2]}
    )

    for i, key1 in enumerate(cat1_keys):
        for j, key2 in enumerate(cat2_keys):
            freq_key = "_".join([key1, key2])
            ax_arr[1].plot(
                range(num_clusters),
                np.squeeze(freq[combined_keys == freq_key, :]),
                label=key1,
                color=colors[-i - 1],
                alpha=0.1,
            )

    handles = [
        Line2D([0], [0], linewidth=5, color=colors[-i - 1], label=cat1_keys[i])
        for i in range(len(cat1_keys))
    ]
    ax_arr[1].spines["top"].set_visible(False)
    ax_arr[1].get_xaxis().set_visible(False)
    ax_arr[1].spines["right"].set_visible(False)
    ax_arr[1].spines["bottom"].set_visible(False)
    ax_arr[1].set_ylabel("% Time Spent in Cluster")
    # TODO: Make this ncol programmable
    ax_arr[1].legend(handles, cat1_keys, loc="upper right", ncol=1, borderpad=1)
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
    j = 1
    for i, key1 in enumerate(cat1_keys):
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
    ax_arr[2].spines["top"].set_visible(False)
    ax_arr[2].spines["right"].set_visible(False)

    ax_arr[2].set_ylabel("Mean")
    ax_arr[2].set_xlabel("Cluster")
    ax_arr[2].set_xlim([-0.25, freq.shape[1] + 0.25])
    f.tight_layout()
    plt.subplots_adjust(hspace=0.05)
    Path(filepath).mkdir(parents=True, exist_ok=True)
    plt.savefig("".join([filepath, "mean_sd_cluster_freq.png"]), dpi=200)
    if show:
        plt.show()

    plt.close()
    return None


def heuristics(
    features: npt.NDArray[Any],
    labels: List[str],
    data_obj: ds.DataStruct,
    heuristics: Dict[str, Dict[str, List[str]]],
    filepath: str,
) -> None:
    """
    Visualize heuristic-derived scores on watershed maps.

    Parameters
    ----------
    features : ndarray, shape (n_frames, n_features)
    labels : list of str
    data_obj : DataStruct
    heuristics : dict
        Mapping heuristic name -> {"high": [...], "low": [...]}
    filepath : str
        Output directory prefix.

    Returns
    -------
    None
    """
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
        except AssertionError:
            print("Couldn't find some features from the heuristics")

        high_feats = np.clip(features[:, high_feat_i], -2.5, 2.5)
        low_feats = np.clip(-features[:, low_feat_i], -2.5, 2.5)
        heur_feats_arr = np.mean(np.append(high_feats, low_feats, axis=1), axis=1)

        heur_watershed = data_obj.ws.watershed_map.copy()
        for cluster in np.unique(data_obj.data["Cluster"].values):
            cluster_mean = np.mean(heur_feats_arr[data_obj.data["Cluster"] == cluster])
            heur_watershed[data_obj.ws.watershed_map == cluster] = cluster_mean

        watershed(
            ws_map=heur_watershed,
            ws_borders=data_obj.ws.borders,
            filepath=filepath + heur_key,
        )

        print("Highest " + heur_key + " score frames: ")
        sorted_idx = np.argsort(heur_feats_arr)
        print(sorted_idx)


def _ax_bubble_map(
    ax: Any,
    embed_vals: npt.ArrayLike,
    borders: npt.ArrayLike,
    unique_annotations: List[str],
    clusters: npt.ArrayLike,
    annotations: npt.ArrayLike,
    radius: Union[str, npt.ArrayLike] = "frequency",
    scale_factor: float = 1000,
    get_legend_handles: bool = True,
) -> Tuple[Any, List[Any]]:
    """
    Draw bubble overlays for clusters on a matplotlib Axes.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
    embed_vals : array-like, shape (N, 2)
    borders : array-like
    unique_annotations : list of str
    clusters : array-like, shape (N,)
    annotations : array-like, shape (N,) or (n_clusters,)
    radius : 'frequency' or array-like, default 'frequency'
    scale_factor : float, default 1000
    get_legend_handles : bool, default True

    Returns
    -------
    ax : matplotlib.axes.Axes
    legend_handles : list
    """
    n_clusters = len(annotations)
    n_vals = len(embed_vals)

    for k, v in borders.items():
        ax.plot(
            v[:, 0],
            v[:, 1],
            color="k",
            markersize=0,
            lw=0.25,
            zorder=1,
        )

    data_arr = np.concatenate([embed_vals, clusters[:, None]], axis=-1)
    data_df = pd.DataFrame(data=data_arr, columns=["x", "y", "cluster"])
    agg_methods = dict(x_mean=("x", "mean"), y_mean=("y", "mean"))
    if isinstance(radius, str):
        if radius == "frequency":
            agg_methods["radius"] = ("cluster", "size")
    else:
        if len(radius) == n_vals:
            agg_methods["radius"] = ("radius", "mean")

    cluster_stats = data_df.groupby("cluster").agg(**agg_methods)
    cluster_stats = cluster_stats.reindex(range(1, n_clusters + 1), fill_value=0)
    cluster_stats["annotations"] = annotations

    if isinstance(radius, str):
        if radius == "frequency":
            cluster_stats["radius"] = cluster_stats["radius"] / n_vals
    else:
        if len(radius) == len(cluster_stats):
            cluster_stats["radius"] = radius

    legend_handles = []
    for idx, row in cluster_stats.iterrows():
        color = PALETTE[unique_annotations.index(row["annotations"])]
        rad = row["radius"] * scale_factor
        circle = plt.Circle(
            (row["x_mean"], row["y_mean"]),
            rad,
            facecolor=color,
            alpha=0.75,
            edgecolor="black",
            linewidth=1.2,
            linestyle="-" if rad >= 0 else "--",
        )
        ax.add_patch(circle)

        if (
            row["annotations"] not in [h.get_label() for h in legend_handles]
        ) & get_legend_handles:
            legend_handles.append(
                plt.Line2D(
                    [0],
                    [0],
                    marker="o",
                    color="w",
                    markerfacecolor=color,
                    markersize=10,
                    label=row["annotations"],
                )
            )

    ax.set_aspect(0.9)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.axis("off")

    return ax, legend_handles


def bubble_map(
    embed_vals: npt.NDArray[np.float64_],
    watershed: Watershed,
    clusters: npt.ArrayLike,
    annotations: Optional[npt.ArrayLike] = None,
    radius: Union[str, npt.ArrayLike] = "frequency",
    scale_factor: float = 1000,
    filepath: Optional[str] = "./bubble_map.svg",
    show_legend: bool = True,
) -> None:
    """
    Create and save a bubble map visualizing cluster centers and sizes.

    Parameters
    ----------
    embed_vals : ndarray, shape (N, 2)
    watershed : Watershed
    clusters : array-like, shape (N,)
    annotations : array-like, optional
    radius : 'frequency' or array-like
    scale_factor : float
    filepath : str
    show_legend : bool

    Returns
    -------
    None
    """
    borders = copy.deepcopy(watershed.borders)
    watershed_shape = watershed.watershed_map.shape
    unique_annotations = list(np.unique(annotations))

    # Convert embed vals into pixel units
    for ind in range(2):
        embed_vals[:, ind] = (
            (embed_vals[:, ind] - watershed.hist_range[ind][0])
            / (watershed.hist_range[ind][1] - watershed.hist_range[ind][0])
            * watershed_shape[ind]
        )

    # Border values in imshow coordinates, flipping y values around to match scatterplot
    for k, v in borders.items():
        borders[k][:, 1] = watershed_shape[1] - v[:, 1]

    f = plt.figure(figsize=(10, 10))
    ax = f.add_subplot(1, 1, 1)
    ax, legend_handles = _ax_bubble_map(
        ax,
        embed_vals=embed_vals,
        borders=borders,
        unique_annotations=unique_annotations,
        clusters=clusters,
        annotations=annotations,
        radius=radius,
        scale_factor=scale_factor,
        get_legend_handles=show_legend,
    )

    if show_legend:
        plt.legend(handles=legend_handles, ncols=4, loc="upper center")
    f.tight_layout()
    f.subplots_adjust(top=0.75, bottom=0.001)
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(filepath, dpi=400)
    return None


def annotated_watershed(
    watershed: Watershed,
    annotations: npt.ArrayLike,
    palette: Optional[List] = None,
    filepath: Optional[str] = "./bubble_map.svg",
    show: bool = False,
) -> None:
    """
    Color a watershed map by annotations and draw merged borders and legend.

    Parameters
    ----------
    watershed : Watershed
    annotations : array-like, shape (n_clusters,) or (n_pixels,)
    palette : list, optional
    filepath : str
    show : bool

    Returns
    -------
    None
    """
    unique_annotations = list(np.unique(annotations))

    if palette is None:
        palette = PALETTE

    watershed_map = watershed.watershed_map
    colored_map = np.ones(watershed_map.shape + (3,))

    # Apply colors to the map based on annotations
    for i, annotation in enumerate(unique_annotations):
        inds = np.where(annotations == annotation)[0] + 1
        is_anno = np.isin(watershed_map, inds)
        colored_map[is_anno] = palette[i]

    f = plt.figure(figsize=(10, 10))
    ax = f.add_subplot(111)
    im = ax.imshow(colored_map, alpha=0.6)
    ax.set_aspect(0.9)

    # Draw thin individual borders
    for k, v in watershed.borders.items():
        ax.plot(v[:, 0], v[:, 1], "k", markersize=0, lw=0.25)

    # Plot thick borders around merged regions with same annotation
    for annotation in unique_annotations:
        inds = np.where(annotations == annotation)[0] + 1
        mask = np.isin(watershed_map, inds)

        contours = measure.find_contours(mask.astype(float), level=0.5)
        for contour in contours:
            ax.plot(contour[:, 1], contour[:, 0], "k", lw=1.25)

    # Outer border around all non-zero regions
    nonzero_mask = watershed_map != 0
    outer_contours = measure.find_contours(nonzero_mask.astype(float), level=0.5)
    for contour in outer_contours:
        ax.plot(contour[:, 1], contour[:, 0], "k", lw=1.5)

    ax.axis("off")

    # ---- Add Legend ----
    legend_elements = [
        Patch(facecolor=palette[i], edgecolor="k", label=label)
        for i, label in enumerate(unique_annotations)
    ]
    ax.legend(
        handles=legend_elements,
        ncols=len(unique_annotations),
        loc="upper center",
        frameon=True,
        prop={"family": "Liberation Sans", "size": 16},
    )

    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(filepath, dpi=400)
    if show:
        plt.show()
    plt.close()
    return None


def feature_watershed(
    watershed: npt.ArrayLike,
    watershed_borders: npt.ArrayLike,
    feature: npt.ArrayLike,
    labels: List[Tuple],
    cluster_labels: Optional[npt.ArrayLike] = None,
    filepath: Optional[str] = "./bubble_map.svg",
    show: bool = True,
) -> None:
    """
    Render feature values on watershed clusters using a diverging colormap.

    Parameters
    ----------
    watershed : ndarray, shape (H, W)
        Integer map of cluster labels per pixel (can be used to index cluster features).
    watershed_borders : dict-like
        Borders mapping for overlay.
    feature : array-like, shape (n_clusters,) or (n_frames,)
        Feature values either per cluster or per frame (if per frame, supply cluster_labels).
    labels : list of tuples
        Not used in computation (kept for compatibility).
    cluster_labels : array-like, shape (n_frames,), optional
        If `feature` is per-frame, this maps frames to clusters.
    filepath : str
    show : bool

    Returns
    -------
    None
    """
    n_clusters = int(np.max(watershed)) + 1
    if len(feature) in [n_clusters, n_clusters - 1]:
        feat = feature
    else:
        feat_dict = {"feature": feature, "cluster": cluster_labels}
        df = pd.DataFrame().assign(**feat_dict)
        feat = df.groupby("cluster").agg({"feature": "mean"})
        assert len(feat) in [n_clusters, n_clusters - 1]

    f = plt.figure()
    ax = f.add_subplot(111)
    im = ax.imshow(
        feat[watershed],
        cmap=CUSTOM_CMAPS["wide_white_seismic"],
        vmin=-5,
        vmax=5,
    )
    ax.set_aspect(0.9)
    for k, v in watershed_borders.items():
        ax.plot(v[:, 0], v[:, 1], "k", markersize=0, lw=0.25)
    ax.axis("off")

    f.colorbar(im)

    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(filepath, dpi=200)
    if show:
        plt.show()
    plt.close()
    return None


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
