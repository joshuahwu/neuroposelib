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

def get_backend(x):
    """Return a backend module with NumPy-like ops:
       np, torch, or tf-like object.
    """
    import numpy as _np
    try:
        import torch as _torch
    except ImportError:
        _torch = None
    try:
        import tensorflow as _tf
    except ImportError:
        _tf = None

    if isinstance(x, _np.ndarray):
        return _np
    if _torch is not None and _torch.is_tensor(x):
        return _torch
    if _tf is not None and isinstance(x, _tf.Tensor):
        return _tf

    raise TypeError("Unsupported type for distortPoints(): {}".format(type(x)))

def distortPoints(points, intrinsicMatrix, radialDistortion, tangentialDistortion):
    """Distort points according to camera parameters.

    Ported from Matlab 2018a
    """
    # unpack the intrinisc matrix
    cx = intrinsicMatrix[2, 0]
    cy = intrinsicMatrix[2, 1]
    fx = intrinsicMatrix[0, 0]
    fy = intrinsicMatrix[1, 1]
    skew = intrinsicMatrix[1, 0]

    # center the points
    center = np.array([cx, cy])
    centeredPoints = points - center[np.newaxis, :]

    # normalize the points
    yNorm = centeredPoints[:, 1] / fy
    xNorm = (centeredPoints[:, 0] - skew * yNorm) / fx

    # compute radial distortion
    r2 = xNorm ** 2 + yNorm ** 2
    r4 = r2 * r2
    r6 = r2 * r4

    k = np.zeros((3,))
    k[:2] = radialDistortion[:2]
    if len(radialDistortion) < 3:
        k[2] = 0
    else:
        k[2] = radialDistortion[2]
    alpha = k[0] * r2 + k[1] * r4 + k[2] * r6

    # compute tangential distortion
    p = tangentialDistortion
    xyProduct = xNorm * yNorm
    dxTangential = 2 * p[0] * xyProduct + p[1] * (r2 + 2 * xNorm ** 2)
    dyTangential = p[0] * (r2 + 2 * yNorm ** 2) + 2 * p[1] * xyProduct

    # apply the distortion to the points
    normalizedPoints = np.stack((xNorm, yNorm)).T
    distortedNormalizedPoints = (
        normalizedPoints
        + normalizedPoints * np.array([alpha, alpha]).T
        + np.stack((dxTangential, dyTangential)).T
    )

    # # convert back to pixels
    distortedPointsX = (
        (distortedNormalizedPoints[:, 0] * fx)
        + cx
        + (skew * distortedNormalizedPoints[:, 1])
    )
    distortedPointsY = distortedNormalizedPoints[:, 1] * fy + cy
    distortedPoints = np.stack((distortedPointsX, distortedPointsY))

    return distortedPoints

def project_to2d(
    pts: np.ndarray, K: np.ndarray, R: np.ndarray, t: np.ndarray
) -> np.ndarray:
    """Project 3d points to 2d.

    Projects a set of 3-D points, pts, into 2-D using the camera intrinsic
    matrix (K), and the extrinsic rotation matric (R), and extrinsic
    translation vector (t). Note that this uses the matlab
    convention, such that
    M = [R;t] * K, and pts2d = pts3d * M
    """

    M = np.concatenate((R, t), axis=0) @ K
    projPts = np.concatenate((pts, np.ones((pts.shape[0], 1))), axis=1) @ M
    projPts[:, :2] = projPts[:, :2] / projPts[:, 2:]

    return projPts