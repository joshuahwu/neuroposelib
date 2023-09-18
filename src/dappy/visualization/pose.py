import os
import numpy as np
import tqdm

from matplotlib.lines import Line2D
import matplotlib

from pathlib import Path

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
from typing import Optional, Union, List, Tuple

from dappy import DataStruct as ds


def sample3D(
    pose: np.ndarray,
    connectivity: ds.Connectivity,
    labels: Union[np.ndarray, List],
    n_samples: int = 9,
    vid_label: str = "cluster",
    centered: bool = True,
    N_FRAMES: int = 100,
    fps: int = 90,
    show_map: bool = True,
    filepath: str = "./plot_folder",
):
    assert pose.shape[0] == len(labels)
    index = np.arange(len(labels))

    unique_labels = np.unique(labels)

    for cat in tqdm.tqdm(unique_labels):
        label_idx = index[labels == cat]
        if len(label_idx) == 0:
            continue
        else:
            num_points = min(len(label_idx), n_samples)
            permuted_points = np.random.permutation(
                label_idx
            )  # b/c moving frames filter
            sampled_points = []
            for i in range(len(permuted_points)):
                if len(sampled_points) == num_points:  # sampled enough points
                    break
                elif any(
                    np.abs(permuted_points[i] - np.array(sampled_points)) < 200
                ):  # point is not far enough from previous points
                    continue
                elif permuted_points[i] < (N_FRAMES / 2):
                    continue
                elif permuted_points[i] > (len(labels) - N_FRAMES / 2):
                    continue
                else:
                    sampled_points += [permuted_points[i]]

            assert np.all(labels[sampled_points] == cat)

            print(sampled_points)
            if show_map:
                pose3D_expanded(
                    pose,
                    label=cat,
                    connectivity=connectivity,
                    frames=sampled_points,
                    N_FRAMES=N_FRAMES,
                    fps=fps,
                    VID_NAME="".join([vid_label, str(cat), ".mp4"]),
                    SAVE_ROOT="".join([filepath, "/skeleton_vids/"]),
                )
            else:
                arena3D(
                    pose,
                    connectivity=connectivity,
                    frames=sampled_points,
                    centered=centered,
                    N_FRAMES=N_FRAMES,
                    fps=fps,
                    VID_NAME="".join([vid_label, str(cat), ".mp4"]),
                    SAVE_ROOT="".join([filepath, "/skeleton_vids/"]),
                )

def arena3D_map(
    pose: np.ndarray,
    connectivity: ds.Connectivity,
    frames: Union[List[int], int] = [3000, 100000, 500000],
    centered: bool = True,
    N_FRAMES: int = 300,
    fps: int = 90,
    dpi: int = 200,
    VID_NAME: str = "0.mp4",
    SAVE_ROOT: str = "./test/pose_vids/",
):
    if isinstance(frames, int):
        frames = [frames]

    pose_3d, limits, links, COLORS = _init_vid3D(
        pose, connectivity, np.array(frames, dtype=int), centered, N_FRAMES, SAVE_ROOT
    )

    # Set up video writer
    writer = FFMpegWriter(fps=fps)
    # Setup figure
    figsize = (12, 12)
    fig = plt.figure(figsize=figsize)
    ax_3d = fig.add_subplot(1, 1, 1, projection="3d")
    with writer.saving(fig, os.path.join(SAVE_ROOT, "vis_" + VID_NAME), dpi=dpi):
        for curr_frame in tqdm.tqdm(range(N_FRAMES)):
            curr_frames = curr_frame + np.arange(len(frames)) * N_FRAMES
            ax_3d = _pose3D_arena(
                ax_3d, pose_3d, COLORS, links, curr_frames, limits, figsize
            )

            # grab frame and write to vid
            writer.grab_frame()
            ax_3d.clear()
        fig.tight_layout

    plt.close()
    return

def arena3D_map(
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
    pose_3d, limits, links_expand, COLOR = _init_vid3D(
        data, connectivity, frames, N_FRAMES, SAVE_ROOT
    )

    # set up video writer
    writer = FFMpegWriter(fps=fps)

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

    with writer.saving(fig, os.path.join(SAVE_ROOT, "vis_" + VID_NAME), dpi=dpi):
        for curr_frame in tqdm.tqdm(range(N_FRAMES)):
            # grab frames
            curr_frames = curr_frame + np.arange(len(frames)) * N_FRAMES
            kpts_3d = np.reshape(
                pose_3d[curr_frames, :, :], (len(frames) * pose_3d.shape[-2], 3)
            )

            # plot 3d moving skeletons
            for i in range(len(np.unique(frames_meta))):
                temp_idx = curr_frames[frames_meta == np.unique(frames_meta)[i]]
                temp_kpts = np.reshape(
                    [pose_3d[temp_idx, :, :]], (len(temp_idx) * pose_3d.shape[-2], 3)
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

            ax_3d.set_xlim(*limits[0, :])
            ax_3d.set_ylim(*limits[1, :])
            ax_3d.set_zlim(*limits[2, :])
            ax_3d.set_xlabel("x")
            ax_3d.set_ylabel("y")
            ax_3d.set_box_aspect(limits[:, 1] - limits[:, 0])

            # grab frame and write to vid
            writer.grab_frame()
            fig.tight_layout()
            ax_3d.clear()

    plt.close()
    return 0


def get_3d_limits(pose: np.ndarray):
    # compute 3d grid limits
    limits = np.append(
        np.min(pose, axis=(0, 1))[:, None],
        np.max(pose, axis=(0, 1))[:, None],
        axis=1,
    )

    distance = (limits[:, 1] - limits[:, 0]) * 0.05
    offset = np.array([-distance, distance]).T
    offset[2, 0] = 0
    limits += offset

    limits[2, 0] = np.minimum(0, limits[2, 0])  # z-min

    return limits


def _pose3D_frame(
    ax_3d: matplotlib.axes.Axes,
    pose: np.ndarray,
    COLOR: np.ndarray,
    links: np.ndarray,
    limits: Optional[np.ndarray] = None,
):
    """
    Plot single pose given a 3D matplotlib.axes.Axes object
    """

    # Plot keypoints
    ax_3d.scatter(
        pose[:, 0],
        pose[:, 1],
        pose[:, 2],
        marker="o",
        color="black",
        s=30,
        alpha=0.5,
    )

    # Plot keypoint segments
    for color, (index_from, index_to) in zip(COLOR, links):
        xs, ys, zs = [
            np.array([pose[index_from, j], pose[index_to, j]]) for j in range(3)
        ]
        ax_3d.plot3D(xs, ys, zs, c=color, lw=4)

    ax_3d.set_xlim(*limits[0, :])
    ax_3d.set_ylim(*limits[1, :])
    ax_3d.set_zlim(*limits[2, :])

    ax_3d.set_box_aspect(limits[:, 1] - limits[:, 0])
    return ax_3d


def _init_vid3D(
    data: np.ndarray,
    connectivity: ds.Connectivity,
    frames: np.ndarray,
    centered: bool = True,
    N_FRAMES: int = 150,
    SAVE_ROOT: str = "./test/pose_vids/",
):
    Path(SAVE_ROOT).mkdir(parents=True, exist_ok=True)

    if centered:
        frames = frames - int(N_FRAMES / 2) + 1

    COLOR = np.moveaxis(
        np.tile(connectivity.colors[..., None], len(frames)), -1, 0
    ).reshape((-1, 4))
    links = connectivity.links
    links_expand = links

    ## Expanding connectivity for each frame to be visualized
    num_joints = np.max(links) + 1
    for i in range(len(frames) - 1):
        next_con = [
            (x + (i + 1) * num_joints, y + (i + 1) * num_joints) for x, y in links
        ]
        links_expand = np.append(links_expand, np.array(next_con), axis=0)

    # get dannce predictions
    pose_3d = []
    for start in frames:
        pose_3d += [data[start : start + N_FRAMES, ...]]

    pose_3d = np.concatenate(pose_3d, axis=0)

    # compute 3d grid limits
    limits = get_3d_limits(pose_3d)
    return pose_3d, limits, links_expand, COLOR


def _pose3D_arena(
    ax_3d: matplotlib.axes.Axes,
    data: np.ndarray,
    COLORS: np.ndarray,
    links: np.ndarray,
    frames: np.ndarray,
    limits: np.ndarray,
    size: Tuple[int],
    title: Optional[str] = None,
):
    (rows, cols) = size
    kpts_3d = np.reshape(data[frames, :, :], (len(frames) * data.shape[-2], 3))

    ax_3d = _pose3D_frame(
        ax_3d, kpts_3d, COLORS, links, limits  # , figsize=(cols * 5, rows * 5)
    )

    if title is not None:
        ax_3d.set_title(title, fontsize=20, y=0.9)

    return ax_3d


def arena3D(
    pose: np.ndarray,
    connectivity: ds.Connectivity,
    frames: Union[List[int], int] = [3000, 100000, 500000],
    centered: bool = True,
    N_FRAMES: int = 300,
    fps: int = 90,
    dpi: int = 200,
    VID_NAME: str = "0.mp4",
    SAVE_ROOT: str = "./test/pose_vids/",
):
    if isinstance(frames, int):
        frames = [frames]

    pose_3d, limits, links, COLORS = _init_vid3D(
        pose, connectivity, np.array(frames, dtype=int), centered, N_FRAMES, SAVE_ROOT
    )

    # Set up video writer
    writer = FFMpegWriter(fps=fps)
    # Setup figure
    figsize = (12, 12)
    fig = plt.figure(figsize=figsize)
    ax_3d = fig.add_subplot(1, 1, 1, projection="3d")
    with writer.saving(fig, os.path.join(SAVE_ROOT, "vis_" + VID_NAME), dpi=dpi):
        for curr_frame in tqdm.tqdm(range(N_FRAMES)):
            curr_frames = curr_frame + np.arange(len(frames)) * N_FRAMES
            ax_3d = _pose3D_arena(
                ax_3d, pose_3d, COLORS, links, curr_frames, limits, figsize
            )

            # grab frame and write to vid
            writer.grab_frame()
            ax_3d.clear()
        fig.tight_layout

    plt.close()
    return


def _pose3D_grid(
    fig: plt.figure,
    data: np.ndarray,
    connectivity: ds.Connectivity,
    frames: np.ndarray,
    limits: np.ndarray,
    size: Tuple[int],
    labels: Optional[List[str]] = None,
):
    (rows, cols) = size
    for i, curr_frame in enumerate(frames):
        temp_kpts = data[curr_frame, :, :]

        ax_3d = fig.add_subplot(rows, cols, i + 1, projection="3d")
        ax_3d = _pose3D_frame(
            ax_3d,
            temp_kpts,
            connectivity.colors,
            connectivity.links,
            limits,
            # TODO: adjust marker and line sizes w/figsize
            # figsize=(cols * 5, rows * 5),
        )

        ax_3d.grid(False)
        ax_3d.axis(False)
        for xyz_ax in [ax_3d.xaxis, ax_3d.yaxis, ax_3d.zaxis]:
            xyz_ax.set_pane_color((1, 1, 1, 0))
            xyz_ax._axinfo["grid"]["color"] = (1, 1, 1, 0)

        if labels is not None:
            ax_3d.set_title(str(labels[i]), fontsize=20, y=0.9)

    return fig


def grid3D(
    pose: np.ndarray,
    connectivity: ds.Connectivity,
    frames: Union[List[int], int] = [3000, 100000, 5000000],
    centered: bool = True,
    labels: Optional[List] = None,
    title: Optional[str] = None,
    N_FRAMES: int = 150,
    fps: int = 90,
    dpi: int = 100,
    VID_NAME: str = "0.mp4",
    SAVE_ROOT: str = "./test/pose_vids/",
):
    if isinstance(frames, int):
        frames = [frames]
    # Reshape pose and other variables
    pose_3d, limits, links, COLOR = _init_vid3D(
        pose, connectivity, np.array(frames, dtype=int), centered, N_FRAMES, SAVE_ROOT
    )

    # Set up video writer
    writer = FFMpegWriter(fps=fps)
    # Set up figure
    rows = int(np.sqrt(len(frames)))
    cols = int(np.ceil(len(frames) / rows))
    # cols = min(4, len(frames))
    # rows = int(len(frames) / 4) + 1
    figsize = (cols * 5, rows * 5)
    fig = plt.figure(figsize=figsize)

    with writer.saving(fig, os.path.join(SAVE_ROOT, "vis_" + VID_NAME), dpi=dpi):
        for curr_frame in tqdm.tqdm(range(N_FRAMES)):
            curr_frames = curr_frame + np.arange(len(frames)) * N_FRAMES
            fig = _pose3D_grid(
                fig,
                pose_3d,
                connectivity,
                curr_frames,
                limits,
                size=(rows, cols),
                labels=labels,
            )

            if title is not None:
                fig.suptitle(title, fontsize=30)

            fig.tight_layout()
            writer.grab_frame()
            fig.clear()

    plt.close()
    return


def feature_hist(feature, label, filepath, range=None):
    plt.hist(feature, bins=1000, range=range, density=True)
    plt.xlabel(label)
    plt.ylabel("Histogram Density")
    if filepath:
        plt.savefig("".join([filepath, label, "_hist.png"]))
    plt.close()
    return


def features3D(
    pose: np.ndarray,
    feature: np.ndarray,
    connectivity: Optional[ds.Connectivity] = None,
    frames: List = [3000],
    N_FRAMES: int = 150,
    fps: int = 90,
    dpi: int = 200,
    VID_NAME: str = "0.mp4",
    SAVE_ROOT: str = "./test/skeleton_vids/",
):
    if isinstance(frames, int):
        frames = [frames]
    # Reshape pose and other variables
    pose_3d, limits, links_expand, COLOR = _init_vid3D(
        pose, connectivity, frames, N_FRAMES, SAVE_ROOT
    )

    # set up video writer
    writer = FFMpegWriter(fps=int(fps / 4))

    # Setup figure
    fig = plt.figure(figsize=(20, 10))
    gs = fig.add_gridspec(1, 2)
    ax_3d = fig.add_subplot(gs[0, 1], projection="3d")
    ax_trace = fig.add_subplot(gs[0, 0])

    with writer.saving(fig, os.path.join(SAVE_ROOT, "vis_feat_" + VID_NAME), dpi=dpi):
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

            ax_3d.set_xlim(x_min, x_max)
            ax_3d.set_ylim(y_min, y_max)
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
