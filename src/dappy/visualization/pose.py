import os
import numpy as np
import tqdm

from matplotlib.lines import Line2D
import matplotlib

from pathlib import Path
import functools

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
from typing import Optional, Union, List, Tuple
from dappy.embed import Watershed
from dappy import DataStruct as ds
from dappy.visualization.constants import PALETTE, EPS, DEFAULT_BONE
from dappy.visualization.plot import _mask_density
import copy


def sample(func):
    @functools.wraps(func)
    def wrapper(
        pose: np.ndarray,
        connectivity: ds.Connectivity,
        labels: Union[np.ndarray, List],
        VID_NAME: str = "cluster",
        centered: bool = True,
        n_samples: int = 9,
        N_FRAMES: int = 100,
        watershed: Optional[Watershed] = None,
        embed_vals: Optional[np.ndarray] = None,
        **kwargs,
    ):
        if pose.shape[0] != len(labels):
            print("Detected labels not the same shape as pose...")
            downsample = int(np.ceil(pose.shape[0] / len(labels)))
            print("Assuming labels downsampled by {}".format(downsample))
            assert 0 <= len(labels) * downsample - pose.shape[0] < downsample
        else:
            downsample = 1
        assert (embed_vals is None) or (embed_vals.shape[0] == len(labels))

        index = np.arange(len(labels)) * downsample
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
                    elif permuted_points[i] > (pose.shape[0] - N_FRAMES / 2):
                        continue
                    else:
                        sampled_points += [permuted_points[i]]

                assert np.all(
                    labels[(np.array(sampled_points) / downsample).astype(int)] == cat
                )

                print(sampled_points)

                sampled_slice = np.add.outer(
                    sampled_points, np.arange(N_FRAMES)
                ).flatten()

                if centered:
                    sampled_slice -= N_FRAMES // 2

                cat_embed_vals = (
                    None if embed_vals is None else embed_vals[labels == cat, :]
                )
                cat_watershed = copy.deepcopy(watershed)
                if cat_watershed is not None:
                    cat_watershed.watershed_map = np.where(
                        watershed.watershed_map == cat, 1, 0.1
                    )
                    cat_watershed.watershed_map = np.where(
                        watershed.watershed_map == 0, 0, cat_watershed.watershed_map
                    )

                func(
                    pose=pose[sampled_slice, ...],
                    connectivity=connectivity,
                    VID_NAME=VID_NAME + str(cat),
                    embed_vals=cat_embed_vals,
                    watershed=cat_watershed,
                    n_samples=num_points,
                    N_FRAMES=N_FRAMES,
                    **kwargs,
                )

    return wrapper


@sample
def sample_arena3D(
    pose: np.ndarray,
    connectivity: ds.Connectivity,
    n_samples: int = 9,
    VID_NAME: str = "cluster",
    N_FRAMES: int = 100,
    watershed: Optional[Watershed] = None,
    embed_vals: Optional[np.ndarray] = None,
    filepath: str = "./plot_folder",
    **kwargs,
):
    if watershed is not None:
        if embed_vals is not None:
            density = watershed.fit_density(
                embed_vals, new=False
            )  # Fit density on old axes
        else:
            density = watershed.watershed_map

        arena3D_map(
            pose=pose,
            density=_mask_density(density, watershed.watershed_map, eps=EPS * 1.01),
            watershed_borders=watershed.borders,
            connectivity=connectivity,
            frames=np.arange(n_samples) * N_FRAMES,
            centered=False,
            N_FRAMES=N_FRAMES,
            VID_NAME=VID_NAME + ".mp4",
            SAVE_ROOT="".join([filepath, "/skeleton_vids/"]),
            **kwargs,
        )
    else:
        arena3D(
            pose=pose,
            connectivity=connectivity,
            frames=np.arange(n_samples) * N_FRAMES,
            centered=False,
            N_FRAMES=N_FRAMES,
            VID_NAME=VID_NAME + ".mp4",
            SAVE_ROOT="".join([filepath, "/skeleton_vids/"]),
            **kwargs,
        )

    return


@sample
def sample_grid3D(
    pose: np.ndarray,
    connectivity: ds.Connectivity,
    n_samples: int = 9,
    VID_NAME: str = "cluster",
    N_FRAMES: int = 100,
    watershed: Optional[Watershed] = None,
    embed_vals: Optional[np.ndarray] = None,
    filepath: str = "./plot_folder",
    **kwargs,
):
    if watershed is not None:
        if embed_vals is not None:
            density = watershed.fit_density(
                embed_vals, new=False
            )  # Fit density on old axes
        else:
            density = watershed.watershed_map

        grid3D_map(
            pose=pose,
            density=_mask_density(density, watershed.watershed_map, eps=EPS * 1.01),
            watershed_borders=watershed.borders,
            connectivity=connectivity,
            frames=np.arange(n_samples) * N_FRAMES,
            centered=False,
            N_FRAMES=N_FRAMES,
            VID_NAME=VID_NAME + ".mp4",
            SAVE_ROOT="".join([filepath, "/skeleton_vids/"]),
            **kwargs,
        )
    else:
        grid3D(
            pose=pose,
            connectivity=connectivity,
            frames=np.arange(n_samples) * N_FRAMES,
            centered=False,
            N_FRAMES=N_FRAMES,
            VID_NAME=VID_NAME + ".mp4",
            SAVE_ROOT="".join([filepath, "/skeleton_vids/"]),
            **kwargs,
        )

    return


def _plot_density(
    ax: matplotlib.axes.Axes, density: np.ndarray, watershed_borders: np.ndarray
):
    ax.imshow(
        density,
        vmin=EPS,
        cmap=DEFAULT_BONE,
    )

    ax.plot(
        watershed_borders[:, 0],
        watershed_borders[:, 1],
        ".k",
        markersize=0.1,
    )

    ax.set_aspect(0.9)
    ax.axis("off")
    return ax


def init_vid3D(func):
    @functools.wraps(func)
    def wrapper(
        pose: np.ndarray,
        connectivity: ds.Connectivity,
        frames: Union[List[int], int] = [3000, 100000, 500000],
        centered: bool = True,
        N_FRAMES: int = 300,
        SAVE_ROOT: str = "./test/pose_vids/",
        **kwargs,
    ):
        if isinstance(frames, int):
            frames = [frames]

        pose_3d, limits, links, COLORS = _init_vid3D(
            pose,
            connectivity,
            np.array(frames, dtype=int),
            centered,
            N_FRAMES,
            SAVE_ROOT,
        )

        func(
            pose=pose_3d,
            limits=limits,
            links=links,
            colors=COLORS,
        )

    return wrapper


@init_vid3D
def arena3D_map(
    pose: np.ndarray,
    density: np.ndarray,
    watershed_borders: np.ndarray,
    frames: Union[List[int], int] = [3000, 100000, 500000],
    N_FRAMES: int = 300,
    fps: int = 90,
    dpi: int = 200,
    VID_NAME: str = "0.mp4",
    SAVE_ROOT: str = "./test/pose_vids/",
    **kwargs,
):
    # Set up video writer
    writer = FFMpegWriter(fps=fps)
    # Setup figure
    figsize = (24, 12)
    fig = plt.figure(figsize=figsize, layout="constrained")
    gs = fig.add_gridspec(1, 2)
    ax_3d = fig.add_subplot(gs[0, 1], projection="3d")
    ax_dens = fig.add_subplot(gs[0, 0])
    ax_dens = _plot_density(ax_dens, density, watershed_borders)

    with writer.saving(fig, os.path.join(SAVE_ROOT, "vis_" + VID_NAME), dpi=dpi):
        for curr_frame in tqdm.tqdm(range(N_FRAMES)):
            curr_frames = curr_frame + np.arange(len(frames)) * N_FRAMES
            ax_3d = _pose3D_arena(
                ax_3d,
                pose,
                kwargs["COLORS"],
                kwargs["links"],
                curr_frames,
                kwargs["limits"],
                figsize,
            )

            # grab frame and write to vid
            writer.grab_frame()
            ax_3d.clear()

    plt.close()
    return


def grid3D_map(
    pose: np.ndarray,
    density: np.ndarray,
    watershed_borders: np.ndarray,
    connectivity: ds.Connectivity,
    frames: Union[List[int], int] = [3000, 100000, 5000000],
    centered: bool = True,
    subtitles: Optional[List] = None,
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
    figsize = (cols * 8, rows * 4)
    fig = plt.figure(figsize=figsize, layout="constrained")
    subfig = fig.subfigures(1, 2)
    # import pdb; pdb.set_trace()
    # gs = fig.add_gridspec(1,2)

    ax_dens = subfig[0].add_subplot(1, 1, 1)
    ax_dens = _plot_density(ax_dens, density, watershed_borders)

    with writer.saving(fig, os.path.join(SAVE_ROOT, "vis_" + VID_NAME), dpi=dpi):
        for curr_frame in tqdm.tqdm(range(N_FRAMES)):
            curr_frames = curr_frame + np.arange(len(frames)) * N_FRAMES

            # ax_dens = fig.add_subplot(rows, cols, 1)
            # ax_dens = _plot_density(ax_dens, density, watershed_borders)
            subfig[1] = _pose3D_grid(
                subfig[1],
                pose_3d,
                connectivity,
                curr_frames,
                limits,
                size=(rows, cols),
                subtitles=subtitles,
            )

            if title is not None:
                subfig[1].suptitle(title, fontsize=30)

            writer.grab_frame()
            subfig[1].clear()

    plt.close()
    return


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
        frames = frames - N_FRAMES // 2

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
    fig = plt.figure(figsize=figsize, layout="constrained")
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

    plt.close()
    return


def _pose3D_grid(
    fig: plt.figure,
    data: np.ndarray,
    connectivity: ds.Connectivity,
    frames: np.ndarray,
    limits: np.ndarray,
    size: Tuple[int],
    subtitles: Optional[List[str]] = None,
):
    (rows, cols) = size
    for i, curr_frame in enumerate(frames):
        temp_kpts = data[curr_frame, :, :]
        # ax_3d = ax_3d[i//cols, i%cols]

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

        if subtitles is not None:
            ax_3d.set_title(str(subtitles[i]), fontsize=20, y=0.9)

    return fig


def grid3D(
    pose: np.ndarray,
    connectivity: ds.Connectivity,
    frames: Union[List[int], int] = [3000, 100000, 5000000],
    centered: bool = True,
    subtitles: Optional[List] = None,
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
    figsize = (cols * 5, rows * 5)
    fig = plt.figure(figsize=figsize, layout="constrained")

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
                subtitles=subtitles,
            )

            if title is not None:
                fig.suptitle(title, fontsize=30)

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
    fig = plt.figure(figsize=(20, 10), layout="constrained")
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
            ax_3d.clear()

    plt.close()
    return 0
