from __future__ import annotations
import functools
from tqdm import tqdm
import numpy as np
from typing import Union, List, Callable, Any, Optional, Tuple
import numpy.typing as npt

ArrayLikeInt = Union[npt.NDArray[np.integer], List[int]]


def format_three_digits(val: float) -> str:
    """Format a float with up to three significant digits for human display.

    The formatting rules are:
    - Values with absolute magnitude >= 100 are rounded to an integer (no decimals).
    - Values with absolute magnitude >= 10 use one decimal place.
    - Smaller values use two decimal places.
    - Zero is formatted as "0.00".

    Parameters
    ----------
    val : float
        Value to format.

    Returns
    -------
    str
        The formatted string representation.
    """
    if val == 0:
        return "0.00"

    abs_val = abs(val)

    if abs_val >= 100:
        # 3 digits before the decimal, no decimals
        val_rounded = round(val)
        return f"{val_rounded:.0f}"
    elif abs_val >= 10:
        # 2 digits before, 1 after
        val_rounded = round(val, 1)
        return f"{val_rounded:.1f}"
    else:
        # 1 digit before (or less), 2 after
        val_rounded = round(val, 2)
        return f"{val_rounded:.2f}"


def by_id(func: Callable[..., npt.NDArray[Any]]) -> Callable[..., npt.NDArray[Any]]:
    """Decorator that applies a function to each id-group in a pose array.

    The wrapped function is called once per unique id value found in ``ids``.
    For each id the decorator extracts the sub-array of frames with that id,
    calls the wrapped function with the sub-array (and any keyword arguments),
    and writes the returned sub-array back into the corresponding frames of
    the original ``pose`` array. The original ``pose`` array is modified
    in-place and also returned.

    Parameters
    ----------
    func : callable
        A function that accepts a pose sub-array of shape
        ``(n_frames_id, n_keypoints, 3)`` and returns an array of the same
        shape. It should accept keyword arguments as needed.

    Returns
    -------
    callable
        A wrapper with signature ``(pose, ids, **kwargs) -> np.ndarray`` where
        ``pose`` is an array of shape ``(n_frames, n_keypoints, 3)`` and
        ``ids`` groups frames by id.

    Notes
    -----
    - The wrapper iterates over ``np.unique(ids)`` in ascending order. For
      large datasets or many ids, consider a more efficient grouping strategy
      if needed.
    - The decorator writes results back into the original ``pose`` array (it
      mutates the input). If you need to preserve the original array,
      provide a copy to the wrapped function call site.
    - The wrapped function must return an array compatible with assignment
      into the slice ``pose[ids == i, :, :]`` (same shape as the slice).

    Examples
    --------
    >>> @by_id
    ... def my_cleaner(pose_chunk, threshold=5):
    ...     # pose_chunk: (n_frames_chunk, n_keypoints, 3)
    ...     return pose_chunk  # cleaned chunk
    ...
    >>> out = my_cleaner(pose, ids, threshold=6)
    """
    @functools.wraps(func)
    def wrapper(pose: npt.NDArray[Any], ids: ArrayLikeInt, **kwargs: Any) -> npt.NDArray[Any]:
        for _, i in enumerate(tqdm(np.unique(ids))):
            pose_exp = pose[ids == i, :, :]
            pose[ids == i, :, :] = func(pose_exp, **kwargs)
        return pose

    return wrapper


def rolling_window(data: npt.NDArray[Any], window: int) -> npt.NDArray[Any]:
    """Return a rolling-window view of a 2D array, padding edges with edge values.

    The function creates a view shaped ``(n_frames, window, n_channels)`` (or
    with axes swapped depending on input shape) that provides windowed slices
    centered on each original frame. The input is padded on both ends with
    the edge values so that output has the same number of "center" frames as
    the input.

    Parameters
    ----------
    data : ndarray
        Input 2-D array with shape ``(n_frames, n_channels)``.
    window : int
        Odd integer specifying the window length. Must be odd.

    Returns
    -------
    ndarray
        A view of shape ``(n_frames, n_channels, window)`` rotated so the time
        axis is first (see implementation). The returned array is a NumPy view
        created with ``as_strided``; modifying it will affect internal memory.

    Raises
    ------
    AssertionError
        If ``window`` is not odd.

    Notes
    -----
    - This implementation uses ``numpy.lib.stride_tricks.as_strided`` to avoid
      copying data. Be careful when modifying the returned array.
    - The function pads the input with the edge values so windows at the
      signal boundaries are well-defined.
    - The returned axes are swapped to match existing caller expectations.
    - Implementation based from [`here`](https://stackoverflow.com/questions/6811183/rolling-window-for-1d-arrays-in-numpy).
    """
    try:
        assert window % 2 == 1
    except AssertionError:
        raise AssertionError("Window size must be odd")

    # Padding frames with the edge values with (window size//2)
    pad = int(np.floor(window / 2))
    d_pad = np.pad(data, ((pad, pad), (0, 0)), mode="edge").T
    shape = d_pad.shape[:-1] + (d_pad.shape[-1] - pad * 2, window)
    strides = d_pad.strides + (d_pad.strides[-1],)

    return np.swapaxes(
        np.lib.stride_tricks.as_strided(d_pad, shape=shape, strides=strides), 0, 1
    )


def get_frame_diff(x: npt.NDArray[Any], time: int, idx_center: bool = True) -> npt.NDArray[Any]:
    """Compute temporal differences for each frame using a symmetric window.

    For each frame this function computes a difference vector using values
    ``time`` frames in the past and ``time`` frames in the future (when
    ``idx_center=True``) so the result is centered on the current frame.
    When ``idx_center=False`` the difference is computed between the current
    frame and the value ``time`` frames in the past.

    Parameters
    ----------
    x : ndarray
        Input array where the first axis indexes time (``(n_frames, ...)``).
    time : int
        Size of the time offset to use when computing differences.
    idx_center : bool, optional
        If ``True`` compute a centered difference ``(next - prev)`` using
        ``time`` frames on each side. If ``False`` compute ``x - prev``.
        Default is ``True``.

    Returns
    -------
    ndarray
        An array of the same shape as ``x`` containing the temporal differences.
    """
    prev_x = np.append(np.repeat(x[None, 0, ...], time, axis=0), x[:-time, ...], axis=0)
    if idx_center:
        next_x = np.append(
            x[time:, ...], np.repeat(x[None, -1, ...], time, axis=0), axis=0
        )
        diff = next_x - prev_x
    else:
        diff = x - prev_x

    return diff


def remove_edge_ids(id: npt.ArrayLike[np.int_], size: int) -> npt.NDArray[np.int_]:
    """Remove `size` frames from the start and end of each id block.

    Useful for trimming boundary effects: for each unique id label the first
    ``size`` and last ``size`` frame indices are removed and the remaining
    indices are concatenated and returned.

    Parameters
    ----------
    id : array-like
        1-D array of id labels for each frame (length ``n_frames``).
    size : int
        Number of frames to remove from the start and end of each id block.

    Returns
    -------
    ndarray
        1-D integer array of frame indices that remain after trimming.

    Raises
    ------
    AssertionError
        If the resulting number of indices does not match the expected count.
    """
    ind = np.arange(len(id))
    unsorted_unique = id[np.sort(np.unique(id, return_index=True)[1])]

    for i, label in enumerate(unsorted_unique):
        if i == 0:
            ind_out = ind[id == label][size:-size]
        else:
            ind_out = np.append(ind_out, ind[id == label][size:-size])

    assert len(ind_out) == len(id) - len(unsorted_unique) * 2 * size

    return ind_out


def standard_scale(
    features: npt.NDArray[Any],
    labels: List[str],
    clip: Optional[float] = None,
) -> Tuple[npt.NDArray[Any], List[str]]:
    """Standardize feature columns and optionally clip extreme values.

    The function subtracts the per-feature mean and divides by the per-feature
    standard deviation. Features with zero standard deviation are removed and
    their corresponding labels are filtered out. If ``clip`` is provided the
    standardized values are clipped to the interval ``[-clip, clip]``.

    Parameters
    ----------
    features : ndarray
        2-D array of shape ``(n_samples, n_features)``.
    labels : list of str
        List of feature labels with length ``n_features``.
    clip : float, optional
        If provided, clip standardized values to ``[-clip, clip]``. Default is
        ``None`` (no clipping).

    Returns
    -------
    tuple
        Tuple ``(features_out, labels_out)`` where ``features_out`` is the
        standardized (and possibly clipped) feature array with all-constant
        columns removed, and ``labels_out`` is the filtered list of labels
        corresponding to the remaining columns.
    """
    features -= features.mean(axis=0)
    feat_std = np.std(features, axis=0)
    features = features[:, feat_std != 0]
    if clip is None:
        features = features / feat_std[feat_std != 0]
    else:
        features = np.clip(features / feat_std[feat_std != 0], -clip, clip)
    labels = [label for i, label in enumerate(labels) if feat_std[i] != 0]

    return features, labels
