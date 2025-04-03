import functools
from tqdm import tqdm
import numpy as np
from typing import Union, List


def by_id(func):
    @functools.wraps(func)
    def wrapper(pose: np.ndarray, ids: Union[np.ndarray, List], **kwargs):
        for _, i in enumerate(tqdm(np.unique(ids))):
            pose_exp = pose[ids == i, :, :]
            pose[ids == i, :, :] = func(pose_exp, **kwargs)
        return pose

    return wrapper


def rolling_window(data: np.ndarray, window: int):
    """
    Returns a view of data windowed (data.shape, window)
    Pads the ends with the edge values

    Implemented based off:
    https://stackoverflow.com/questions/6811183/rolling-window-for-1d-arrays-in-numpy
    """
    try:
        assert window % 2 == 1
    except ValueError:
        print("Window size must be odd")
        raise

    # Padding frames with the edge values with (window size/2 - 1)
    pad = int(np.floor(window / 2))
    d_pad = np.pad(data, ((pad, pad), (0, 0)), mode="edge").T
    shape = d_pad.shape[:-1] + (d_pad.shape[-1] - pad * 2, window)
    strides = d_pad.strides + (d_pad.strides[-1],)

    return np.swapaxes(
        np.lib.stride_tricks.as_strided(d_pad, shape=shape, strides=strides), 0, 1
    )


def get_frame_diff(x: np.ndarray, time: int, idx_center: bool = True):
    """
    IN:
        x: Numpy array where first axis is time
        time: Size of window to calculate
        idx_center: if `True`, calculates diff centered around point (idx+time - idx-time),
                    if `False`, calculates diff as time before
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


def remove_edge_ids(id: np.array, size: int):
    ind = np.arange(len(id))
    unsorted_unique = id[np.sort(np.unique(id, return_index=True)[1])]

    for i, label in enumerate(unsorted_unique):
        if i == 0:
            ind_out = ind[id == label][size:-size]
        else:
            ind_out = np.append(ind_out, ind[id == label][size:-size])

    assert len(ind_out) == len(id) - len(unsorted_unique) * 2 * size

    return ind_out


def standard_scale(features, labels, clip=None):
    features -= features.mean(axis=0)
    feat_std = np.std(features, axis=0)
    features = features[:, feat_std != 0]
    if clip is None:
        features = features / feat_std[feat_std != 0]
    else:
        features = np.clip(features / feat_std[feat_std != 0], -clip, clip)
    labels = [label for i, label in enumerate(labels) if feat_std[i] != 0]

    return features, labels
