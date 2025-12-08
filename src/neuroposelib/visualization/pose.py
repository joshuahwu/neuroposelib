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
from typing import Optional, Union, List, Tuple, Any, Dict
from neuroposelib.embed import Watershed
from neuroposelib import DataStruct as ds
from neuroposelib.visualization.constants import PALETTE, EPS, DEFAULT_BONE, _PLANE
from neuroposelib.visualization.plot import _mask_density
import copy
import numpy.typing as npt


def sample(func):
    """
    Decorator to sample frames per label and call the wrapped video-creation function
    for each label.

    The decorated function will be invoked once per unique label in `labels` with
    a temporally contiguous slice of `pose` frames for each sampled point.

    Parameters
    ----------
    func : Callable
        Function to decorate. Expected signature (pose, connectivity, labels, ...).

    Returns
    -------
    wrapper : Callable
        Wrapped function that accepts the same args plus sampling arguments.

    The wrapper accepts:
    --------------------
    pose : ndarray, shape (n_frames, n_keypts, 3)
        Full pose array from which sampled contiguous windows will be taken.
    connectivity : ds.Connectivity
        Connectivity information (links, colors, etc.).
    labels : ndarray or list-like, shape (n_labels,)
        Per-frame or per-downsampled-frame labels used to group frames.
    VID_NAME : str
        Base name for output videos.
    centered : bool
        If True, sample windows are centered on the chosen frame (subtract half N_FRAMES).
    n_samples : int
        Number of sampled windows per label to produce.
    N_FRAMES : int
        Number of contiguous frames per sampled window.
    watershed : Watershed or None
        Optional watershed object used to render per-label watershed masks.
    embed_vals : ndarray or None
        Optional embedding values aligned to labels (used to compute density).
    **kwargs : dict
        Extra args forwarded to the wrapped function.
    """

    @functools.wraps(func)
    def wrapper(
        pose: npt.NDArray[np.float_],
        connectivity: ds.Connectivity,
        labels: npt.ArrayLike,
        VID_NAME: str = "cluster",
        centered: bool = True,
        n_samples: int = 9,
        N_FRAMES: int = 100,
        watershed: Optional[Watershed] = None,
        embed_vals: Optional[npt.NDArray[Any]] = None,
        **kwargs: Any,
    ) -> None:
        """
        Wrapper produced by `sample`.

        Parameters
        ----------
        pose : ndarray (n_frames, n_keypts, 3)
            Full pose array aligned to the (possibly downsampled) `labels`.
        connectivity : ds.Connectivity
            Skeleton connectivity settings.
        labels : array-like (n_frames,)
            Labels aligned to frames or to downsampled frames.
        VID_NAME : str
            Base video file name used by inner function.
        centered : bool
            Whether to center the sampled window around the chosen frame.
        n_samples : int
            Number of windows to sample per label.
        N_FRAMES : int
            Window length (frames) for each sampled sample.
        watershed : Watershed or None
            Optional watershed used to create per-label maps.
        embed_vals : ndarray or None
            Optional embedding values aligned to `labels`.
        **kwargs : dict
            Passed through to the decorated function.
        """
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
                permuted_points = np.random.permutation(label_idx)
                sampled_points: List[int] = []
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
    pose: npt.NDArray[np.float_],
    connectivity: ds.Connectivity,
    n_samples: int = 9,
    VID_NAME: str = "cluster",
    N_FRAMES: int = 100,
    watershed: Optional[Watershed] = None,
    embed_vals: Optional[npt.NDArray[Any]] = None,
    filepath: str = "./plot_folder",
    **kwargs: Any,
) -> None:
    """
    Create several small arena-style 3D videos sampled per label.

    Parameters
    ----------
    pose : ndarray, shape (n_frames, n_keypts, 3)
        Full pose array from which short windows will be sampled.
    connectivity : ds.Connectivity
        Skeleton connectivity settings (links, colors).
    n_samples : int
        Number of sample windows (per label) used to choose windows.
    VID_NAME : str
        Base filename (decorator appends label).
    N_FRAMES : int
        Length of each sampled window in frames.
    watershed : Watershed or None
        If given, will create density overlays using `watershed`.
    embed_vals : ndarray or None
        Embedding values aligned to labels. If provided, used to compute density.
    filepath : str
        Root folder where skeleton videos will be saved.
    **kwargs : dict
        Passed through to `arena3D_map` or `arena3D`.
    """
    if watershed is not None:
        if embed_vals is not None:
            density = watershed.fit_density(embed_vals, new=False)
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

    return None


@sample
def sample_grid3D(
    pose: npt.NDArray[np.float_],
    connectivity: ds.Connectivity,
    n_samples: int = 9,
    VID_NAME: str = "cluster",
    N_FRAMES: int = 100,
    watershed: Optional[Watershed] = None,
    embed_vals: Optional[npt.NDArray[np.float_]] = None,
    filepath: str = "./plot_folder",
    **kwargs: Any,
) -> None:
    """
    Create several small grid-style 3D videos sampled per label.

    This function is wrapped by the `@sample` decorator which handles iterating
    over labels and calling this function once per label with sampled windows.

    Parameters
    ----------
    pose : ndarray, shape (n_frames, n_keypts, 3)
        Full 3D pose array from which windows will be sampled.
    connectivity : ds.Connectivity
        Connectivity object that contains `.links` and `.colors`.
    n_samples : int, optional
        Number of windows to sample per label (default 9).
    VID_NAME : str, optional
        Base name for output videos; the decorator appends the label (default "cluster").
    N_FRAMES : int, optional
        Window length (frames) for each sampled sample (default 100).
    watershed : Watershed or None, optional
        If provided, used to compute density overlays for each label.
    embed_vals : ndarray or None, optional
        Embedding values aligned to labels (used to compute density if provided).
    filepath : str, optional
        Base path where outputs are saved (default "./plot_folder").
    **kwargs : dict, optional
        Extra keyword arguments forwarded to `grid3D_map` or `grid3D`.

    Returns
    -------
    None
    """
    if watershed is not None:
        if embed_vals is not None:
            density = watershed.fit_density(embed_vals, new=False)
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

    return None


def _plot_density(
    ax: matplotlib.axes.Axes,
    density: npt.NDArray[np.float_],
    watershed_borders: Dict[int, npt.NDArray[np.float_]],
) -> matplotlib.axes.Axes:
    """
    Helper to render density + borders on an axes.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes on which density will be drawn.
    density : ndarray, shape (H, W)
        Density image to show (already masked/clipped as desired).
    watershed_borders : dict(int -> ndarray)
        Dictionary mapping cluster id to border coordinates.
    """
    ax.imshow(
        density,
        vmin=EPS,
        cmap=DEFAULT_BONE,
    )

    for k, v in watershed_borders.items():
        ax.plot(v[:, 0], v[:, 1], "k", markersize=0, lw=0.25)

    ax.set_aspect(0.9)
    ax.axis("off")
    return ax


def arena3D_map(
    pose: npt.NDArray[np.float_],
    density: npt.NDArray[np.float_],
    watershed_borders: Dict[int, npt.NDArray[np.float_]],
    connectivity: ds.Connectivity,
    frames: Union[List[int], int] = [3000, 100000, 500000],
    centered: bool = True,
    N_FRAMES: int = 300,
    fps: int = 90,
    dpi: int = 200,
    VID_NAME: str = "0.mp4",
    SAVE_ROOT: str = "./test/pose_vids/",
) -> None:
    """
    Create an arena-style 3D video with density panel and 3D pose.

    Parameters
    ----------
    pose : ndarray, shape (n_frames, n_keypts, 3)
        Full 3D pose array used to produce moving skeleton panel.
    density : ndarray, shape (H, W)
        Density image to render in the left panel.
    watershed_borders : dict
        Borders mapping used to overlay lines on density.
    connectivity : ds.Connectivity
        Skeleton connectivity (links, colors).
    frames : list or int
        Frame indices (or single index) used to create windows.
    centered : bool
        Whether to center windows around each frame.
    N_FRAMES : int
        Number of frames in each window (loop length).
    fps : int
        Frames per second for output video.
    dpi : int
        Resolution when writing the video.
    VID_NAME : str
        Output filename for the created video.
    SAVE_ROOT : str
        Directory where video will be saved.
    """
    if isinstance(frames, int):
        frames = [frames]

    pose_3d, limits, links, COLORS = _init_vid3D(
        pose, connectivity, np.array(frames, dtype=int), centered, N_FRAMES, SAVE_ROOT
    )

    # Set up video writer
    writer = FFMpegWriter(fps=fps)
    # Setup figure
    figsize = (24, 12)
    fig = plt.figure(figsize=figsize, layout="constrained")
    gs = fig.add_gridspec(1, 2)
    ax_3d = fig.add_subplot(gs[0, 1], projection="3d")
    ax_dens = fig.add_subplot(gs[0, 0])
    ax_dens = _plot_density(ax_dens, density, watershed_borders)

    Path(SAVE_ROOT).mkdir(parents=True, exist_ok=True)
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
    return None


def grid3D_map(
    pose: npt.NDArray[np.float_],
    density: npt.NDArray[np.float_],
    watershed_borders: Dict[int, npt.NDArray[np.float_]],
    connectivity: ds.Connectivity,
    frames: Union[List[int], int] = [3000, 100000, 5000000],
    centered: bool = True,
    subtitles: Optional[List] = None,
    title: Optional[str] = None,
    N_FRAMES: int = 150,
    fps: int = 90,
    dpi: int = 100,
    figsize: Optional[Tuple[int]] = None,
    VID_NAME: str = "0.mp4",
    SAVE_ROOT: str = "./test/pose_vids/",
) -> None:
    """
    Create a grid-style 3D video with a density panel and a grid of 3D poses.

    Parameters
    ----------
    pose : ndarray, shape (n_frames, n_keypts, 3)
        Full 3D pose array used to construct windows shown in the grid.
    density : ndarray, shape (H, W)
        Density image (e.g. watershed density) to show in the left panel.
    watershed_borders : dict
        Mapping from cluster id to border coordinates for overlaying borders on the density.
    connectivity : ds.Connectivity
        Skeleton connectivity information (links, colors, keypoint colors).
    frames : list of int or int, optional
        Frame indices to use as the reference time for each window; a single int will be converted to a one-element list.
        Default examples: [3000, 100000, 5000000].
    centered : bool, optional
        If True, windows are centered around each frame (subtracts N_FRAMES//2), otherwise windows start at the frame.
    subtitles : list of str or None, optional
        Optional subtitles for each grid panel.
    title : str or None, optional
        Optional overall title placed above the grid of poses.
    N_FRAMES : int, optional
        Number of frames in each window (loop length) used to create the animation (default 150).
    fps : int, optional
        Frames-per-second for the output video (default 90).
    dpi : int, optional
        Resolution used when saving the video (default 100).
    figsize : tuple or None, optional
        Figure size passed to matplotlib. If None, computed from number of frames.
    VID_NAME : str, optional
        Output filename for the created video (default "0.mp4").
    SAVE_ROOT : str, optional
        Directory in which the video file will be saved (default "./test/pose_vids/").

    Returns
    -------
    None
    """
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
    figsize = (cols * 8, rows * 4) if figsize is None else figsize
    fig = plt.figure(figsize=figsize, layout="constrained")
    subfig = fig.subfigures(1, 2)
    # import pdb; pdb.set_trace()
    # gs = fig.add_gridspec(1,2)

    ax_dens = subfig[0].add_subplot(1, 1, 1)
    ax_dens = _plot_density(ax_dens, density, watershed_borders)

    Path(SAVE_ROOT).mkdir(parents=True, exist_ok=True)
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
    return None


def get_3d_limits(pose: npt.NDArray[np.float_]) -> npt.NDArray[np.float_]:
    """
    Compute 3D plotting limits for pose array.

    Parameters
    ----------
    pose : ndarray, shape (n_frames, n_keypts, 3)
        Input pose values.

    Returns
    -------
    limits : ndarray, shape (3, 2)
        For each axis (x,y,z) the [min, max] values padded slightly.
    """
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
    pose: npt.NDArray[np.float_],
    COLOR: npt.NDArray[np.float_],
    links: npt.NDArray[np.int_],
    limits: Optional[npt.NDArray[np.float_]] = None,
) -> matplotlib.axes.Axes:
    """
    Plot a single 3D skeleton pose on the provided Axes3D.

    Parameters
    ----------
    ax_3d : matplotlib.axes.Axes
        3D axes on which to draw.
    pose : ndarray, shape (n_keypts, 3)
        Single frame of 3D keypoints.
    COLOR : ndarray, shape (n_links, 4)
        RGBA colors for link segments.
    links : ndarray, shape (n_links, 2)
        Array of (from, to) keypoint indices for segments.
    limits : ndarray, shape (3,2), optional
        Axis limits (min,max) for each of x,y,z. If None, limits are not set.
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

    if limits is not None:
        ax_3d.set_xlim(*limits[0, :])
        ax_3d.set_ylim(*limits[1, :])
        ax_3d.set_zlim(*limits[2, :])
        ax_3d.set_box_aspect(limits[:, 1] - limits[:, 0])
    return ax_3d


def _init_vid3D(
    data: npt.NDArray[np.float_],
    connectivity: ds.Connectivity,
    frames: npt.NDArray[np.int_],
    centered: bool = True,
    N_FRAMES: int = 150,
    SAVE_ROOT: str = "./test/pose_vids/",
) -> Tuple[npt.NDArray[Any], npt.NDArray[Any], npt.NDArray[Any], npt.NDArray[Any]]:
    """
    Prepare stacked pose windows, plotting limits, expanded links and color array.

    Parameters
    ----------
    data : ndarray, shape (n_frames, n_keypts, 3)
        Full pose array.
    connectivity : ds.Connectivity
        Contains `.links` and `.colors`.
    frames : ndarray of int
        Frame indices for which windows will be prepared.
    centered : bool
        If True, frames are shifted by -N_FRAMES//2 to center windows.
    N_FRAMES : int
        Window length in frames.
    SAVE_ROOT : str
        Directory to ensure exists before saving.

    Returns
    -------
    pose_3d : ndarray, shape (len(frames)*N_FRAMES, n_keypts, 3)
        Concatenated windows of poses.
    limits : ndarray, shape (3,2)
        Plotting limits calculated from concatenated windows.
    links_expand : ndarray, shape (n_links*len(frames), 2)
        Expanded links across concatenated windows.
    COLOR : ndarray, shape (n_links*len(frames), 4)
        Per-link RGBA colors expanded over windows.
    """
    Path(SAVE_ROOT).mkdir(parents=True, exist_ok=True)

    if centered:
        frames = frames - N_FRAMES // 2

    COLOR = np.moveaxis(
        np.tile(connectivity.colors[..., None], len(frames)), -1, 0
    ).reshape((-1, 4))
    links = connectivity.links
    links_expand = links.copy()

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
    data: npt.NDArray[np.float_],
    COLORS: npt.NDArray[np.float_],
    links: npt.NDArray[np.int_],
    frames: npt.NDArray[np.int_],
    limits: npt.NDArray[np.float_],
    size: Tuple[int],
    title: Optional[str] = None,
) -> matplotlib.axes.Axes:
    """
    Plot multiple frames concatenated as a single long skeleton (arena view).

    Parameters
    ----------
    ax_3d : Axes3D
        Axes used for plotting the arena-style concatenated skeleton.
    data : ndarray, shape (total_frames, n_keypts, 3)
        Pose data that will be indexed by `frames`.
    COLORS : ndarray
        Per-link colors, possibly expanded for concatenation.
    links : ndarray, shape (n_links, 2)
        Link definitions.
    frames : ndarray of indices
        Indices selecting frames from `data` (local to concatenated block).
    limits : ndarray, shape (3,2)
        Axis limits.
    size : (rows, cols)
        Used to compute internal layout; kept for compatibility.
    title : str or None
        Optional title to display on the axes.
    """
    (rows, cols) = size
    try:
        kpts_3d = np.reshape(data[frames, :, :], (len(frames) * data.shape[-2], 3))
    except Exception:
        import pdb

        pdb.set_trace()

    ax_3d = _pose3D_frame(
        ax_3d, kpts_3d, COLORS, links, limits  # , figsize=(cols * 5, rows * 5)
    )
    ax_3d.xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax_3d.yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax_3d.zaxis.set_pane_color((0.75, 0.75, 0.75, 0.75))
    """
    ax_3d.w_xaxis.line.set_lw(0.)
    ax_3d.w_yaxis.line.set_lw(0.)
    ax_3d.w_zaxis.line.set_lw(0.)
    """
    for axis in [ax_3d.xaxis, ax_3d.yaxis, ax_3d.zaxis]:
        axis.line.set_linewidth(0.0)
    ax_3d.grid(False)
    ax_3d.set_xticks([])
    ax_3d.set_yticks([])
    ax_3d.set_zticks([])

    if title is not None:
        ax_3d.set_title(title, fontsize=20, y=0.9)

    return ax_3d


def arena3D(
    pose: npt.NDArray[np.float_],
    connectivity: ds.Connectivity,
    frames: Union[List[int], int] = [3000, 100000, 500000],
    centered: bool = True,
    N_FRAMES: int = 300,
    fps: int = 90,
    dpi: int = 200,
    VID_NAME: str = "0.mp4",
    SAVE_ROOT: str = "./test/pose_vids/",
) -> None:
    """
    Create a 3D arena video (single 3D axes) for the given frames.

    Parameters
    ----------
    pose : ndarray (n_frames, n_keypts, 3)
    connectivity : ds.Connectivity
    frames : list or int
        Frames used to generate windows for the video.
    centered, N_FRAMES, fps, dpi, VID_NAME, SAVE_ROOT : various
    """
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
    Path(SAVE_ROOT).mkdir(parents=True, exist_ok=True)
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
    return None


def _pose3D_grid(
    fig: plt.Figure,
    data: npt.NDArray[np.float_],
    connectivity: ds.Connectivity,
    frames: npt.NDArray[np.int_],
    limits: npt.NDArray[np.int_],
    size: Tuple[int],
    subtitles: Optional[List[str]] = None,
) -> plt.Figure:
    """
    Populate a matplotlib Figure with a grid of 3D pose subplots.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        Figure to which subplots will be added.
    data : ndarray (total_frames, n_keypts, 3)
    connectivity : ds.Connectivity
    frames : ndarray of indices
        Local indices into `data` selecting which frames to plot.
    limits : ndarray, shape (3,2)
        Axis limits to apply to each subplot.
    size : tuple (rows, cols)
        Grid dimensions.
    subtitles : list of str or None
        Optional per-panel subtitles.
    """
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
    pose: npt.NDArray[np.float_],
    connectivity: ds.Connectivity,
    frames: Union[List[int], int] = [3000, 100000, 5000000],
    centered: bool = True,
    subtitles: Optional[List[str]] = None,
    title: Optional[str] = None,
    N_FRAMES: int = 150,
    fps: int = 90,
    dpi: int = 100,
    figsize: Optional[Tuple[int]] = None,
    VID_NAME: str = "0.mp4",
    SAVE_ROOT: str = "./test/pose_vids/",
) -> None:
    """
    Create a grid-style 3D video of poses for the provided frames (no density panel).

    Parameters
    ----------
    pose : ndarray, shape (n_frames, n_keypts, 3)
        Full 3D pose array.
    connectivity : ds.Connectivity
        Skeleton connectivity containing `.links` and `.colors`.
    frames : list of int or int, optional
        Frame indices to create windows for. A single int will be converted to a single-element list.
    centered : bool, optional
        If True, windows are centered around each frame (default True).
    subtitles : list of str or None, optional
        Optional per-panel subtitles to annotate each subplot.
    title : str or None, optional
        Optional overall title for the figure.
    N_FRAMES : int, optional
        Number of frames per window (default 150).
    fps : int, optional
        Frames per second for output video (default 90).
    dpi : int, optional
        DPI used when saving the file (default 100).
    figsize : tuple or None, optional
        Figure size to use. If None, computed from grid dimensions.
    VID_NAME : str, optional
        Output filename (default "0.mp4").
    SAVE_ROOT : str, optional
        Directory where the resulting video will be saved.

    Returns
    -------
    None
    """
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
    figsize = (cols * 5, rows * 5) if figsize is None else figsize
    fig = plt.figure(figsize=figsize, layout="constrained")

    Path(SAVE_ROOT).mkdir(parents=True, exist_ok=True)
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
    return None


def feature_hist(
    feature: npt.ArrayLike,
    label: str,
    filepath: str,
    range: Optional[Tuple[float, float]] = None,
) -> None:
    """
    Save a histogram (density) of a single feature.

    Parameters
    ----------
    feature : array-like, shape (N,)
        1D array with feature values.
    label : str
        Label/name used for the saved file and x-axis.
    filepath : str
        Path prefix or full filename where histogram will be saved. Directory is created if needed.
    range : tuple(float, float) or None
        Range to use for histogram binning. If None, automatic range is used.
    """
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    plt.hist(feature, bins=1000, range=range, density=True)
    plt.xlabel(label)
    plt.ylabel("Histogram Density")
    if filepath:
        plt.savefig("".join([filepath, label, "_hist.png"]))
    plt.close()
    return None

# def features3D(
#     pose: np.ndarray,
#     feature: np.ndarray,
#     connectivity: Optional[ds.Connectivity] = None,
#     frames: List = [3000],
#     N_FRAMES: int = 150,
#     fps: int = 90,
#     dpi: int = 200,
#     VID_NAME: str = "0.mp4",
#     SAVE_ROOT: str = "./test/skeleton_vids/",
# ):
#     if isinstance(frames, int):
#         frames = [frames]
#     # Reshape pose and other variables
#     pose_3d, limits, links_expand, COLOR = _init_vid3D(
#         pose, connectivity, frames, N_FRAMES, SAVE_ROOT
#     )

#     # set up video writer
#     writer = FFMpegWriter(fps=int(fps / 4))

#     # Setup figure
#     fig = plt.figure(figsize=(20, 10), layout="constrained")
#     gs = fig.add_gridspec(1, 2)
#     ax_3d = fig.add_subplot(gs[0, 1], projection="3d")
#     ax_trace = fig.add_subplot(gs[0, 0])

#     with writer.saving(fig, os.path.join(SAVE_ROOT, "vis_feat_" + VID_NAME), dpi=dpi):
#         for curr_frame in tqdm.tqdm(range(N_FRAMES)):
#             # grab frames
#             curr_frames = curr_frame + np.arange(len(frames)) * N_FRAMES

#             ax_trace.plot(
#                 np.arange(curr_frames + 1),
#                 feature[: curr_frames[0] + 1],
#                 linestyle="-",
#                 linewidth=1,
#             )
#             ax_trace.plot(
#                 curr_frames, feature[curr_frames], marker=".", markersize=20, color="k"
#             )

#             kpts_3d = np.reshape(
#                 pose_3d[curr_frames, :, :], (len(frames) * num_joints, 3)
#             )

#             # plot 3d moving skeletons
#             ax_3d.scatter(
#                 kpts_3d[:, 0],
#                 kpts_3d[:, 1],
#                 kpts_3d[:, 2],
#                 marker=".",
#                 color="black",
#                 linewidths=0.5,
#             )
#             for color, (index_from, index_to) in zip(COLOR, links_expand):
#                 xs, ys, zs = [
#                     np.array([kpts_3d[index_from, j], kpts_3d[index_to, j]])
#                     for j in range(3)
#                 ]
#                 ax_3d.plot3D(xs, ys, zs, c=color, lw=2)

#             ax_3d.set_xlim(x_min, x_max)
#             ax_3d.set_ylim(y_min, y_max)
#             ax_3d.set_zlim(0, 150)
#             ax_3d.set_xlabel("x")
#             ax_3d.set_ylabel("y")
#             # ax_3d.set_xticks([])
#             # ax_3d.set_yticks([])
#             # ax_3d.set_zticks([])
#             # ax_3d.set_title("3D Tracking")
#             # ax_3d.set_aspect('equal')
#             ax_3d.set_box_aspect([1, 1, 0.4])

#             # grab frame and write to vid
#             writer.grab_frame()
#             ax_3d.clear()

#     plt.close()
#     return 0


def trace(
    pose: npt.ArrayLike,
    connectivity: ds.Connectivity,
    vis_plane: str = "xz",
    frame: int = 1000,
    n_full_pose: int = 3,
    vector: Union[Tuple[int, int], npt.ArrayLike] = (4, 3),
    centered: bool = True,
    N_FRAMES: int = 300,
    dpi: int = 200,
    FIG_NAME: str = "pose_trace.png",
    SAVE_ROOT: str = "./test/pose_vids/",
) -> None:
    """
    Plot a 2D trace (projected plane) of a moving skeleton around a selected frame.

    Parameters
    ----------
    pose : array-like, shape (n_frames, n_keypts, 3)
        Full 3D pose array.
    connectivity : ds.Connectivity
        Connectivity object with `.links` and `.keypt_colors`.
    vis_plane : str, default "xz"
        Two-letter string specifying the plane to visualize (combination of x,y,z).
        Examples: "xy", "xz", "yz".
    frame : int, default 1000
        Central frame index to visualize.
    n_full_pose : int, default 3
        Number of discrete full poses to draw along the trace (visualizes progression).
    vector : tuple or array-like of ints, default (4, 3)
        Two keypoint indices (root_index, forward_index) used to orient/normalize heading.
    centered : bool, default True
        Whether the plotted windows are centered around `frame`.
    N_FRAMES : int, default 300
        Length of the temporal window used when centering/rotating.
    dpi : int, default 200
        Resolution when writing the resulting figure.
    FIG_NAME : str, default "pose_trace.png"
        Filename component to use when saving.
    SAVE_ROOT : str, default "./test/pose_vids/"
        Directory to save the figure to.

    Returns
    -------
    None
    """
    frames = [frame]
    figsize = (5, 5)
    n_keypts = pose.shape[-2]
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(1, 1, 1)
    pose_vis, _, _, _ = _init_vid3D(
        pose, connectivity, np.array(frames, dtype=int), centered, N_FRAMES, SAVE_ROOT
    )
    pose_rot = pose_vis.reshape((len(frames), N_FRAMES, -1, 3))
    pose_rot[..., :2] -= pose_rot[:, N_FRAMES // 2, vector[0], :2][
        :, None, None, :
    ]  # Centering based on middle frame

    forward = (
        pose_rot[:, N_FRAMES // 2, vector[1], :]
        - pose_rot[:, N_FRAMES // 2, vector[0], :]
    )
    forward = forward / np.linalg.norm(forward, axis=-1)[..., None]
    yaw = np.arctan2(forward[:, 1], forward[:, 0])

    # yaw = -get_frame_yaw(pose_rot[:, N_FRAMES // 2, ...], root, 3)
    len_yaw = len(yaw)
    rot_mat = np.array(
        [
            [np.cos(yaw), -np.sin(yaw), np.zeros(len_yaw)],
            [np.sin(yaw), np.cos(yaw), np.zeros(len_yaw)],
            [np.zeros(len_yaw), np.zeros(len_yaw), np.ones(len_yaw)],
        ],
        dtype=pose.dtype,
    )
    # rot_mat = rot_mat.repeat(n_keypts, axis=-1)
    pose_rot = np.einsum("ijkl,lmi->ijkm", pose_rot, rot_mat)

    # if vis_plane == "auto":
    #     dim_std = pose_rot.std(axis=(1, 2)).squeeze()
    #     plane_idx = np.argsort(dim_std, axis=-1)[:, :-1]
    #     # plane_idx = np.zeros(pose_rot.shape,dtype=int)[...,:-1] + plane_idx[:, None, None, :]
    # else:
    plane_idx = [_PLANE[k] for k in vis_plane]

    pose_vis = pose_rot.reshape(-1, n_keypts, 3)
    pose_vis = pose_vis[..., plane_idx]
    print(np.max(pose_vis, axis=(0, 1)))
    print(np.min(pose_vis, axis=(0, 1)))

    full_pose_inds = np.linspace(0, N_FRAMES - 1, n_full_pose).astype(int)
    print(full_pose_inds)
    for i in full_pose_inds:
        # print(i)
        alpha = 1 if i == full_pose_inds[-1] else 0.1
        # print(alpha)
        curr_frames = i + np.arange(len(frames)) * N_FRAMES

        for index_from, index_to in connectivity.links:
            xs, ys = [
                np.array(
                    [
                        pose_vis[curr_frames, index_from, j],
                        pose_vis[curr_frames, index_to, j],
                    ]
                )
                for j in range(2)
            ]
            # lw_color = np.sqrt(np.linspace(0, 0.75, 10))
            # linewidth = 3.5 - np.linspace(0, 3.1, 10)
            # for co, l in zip(lw_color, linewidth):
            ax.plot(
                xs,
                ys,
                # c="k",
                c=(0.1, 0.1, 0.1),
                lw=1,
                alpha=alpha,  # - (i * 0.55 / full_pose_inds[-1])
            )

        ax.scatter(
            pose_vis[curr_frames, :, 0].flatten(),
            pose_vis[curr_frames, :, 1].flatten(),
            marker="o",
            color=np.tile(connectivity.keypt_colors, (len(frames), 1)),
            s=20,
            alpha=alpha,  # 1 - (i * 0.75 / full_pose_inds[-1]),
            zorder=3.5,
        )

    ax.set_ylim(bottom=-50, top=120)
    ax.set_xlim(left=-150, right=150)
    ax.set_aspect("equal")
    ax.axis("off")
    Path(SAVE_ROOT).mkdir(parents=True, exist_ok=True)
    plt.savefig("{}/vis_{}_{}".format(SAVE_ROOT, vis_plane, FIG_NAME), dpi=dpi)
    if os.environ.get("DISPLAY"):
        plt.show()
    plt.close()

    return None
