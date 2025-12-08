import h5py
import numpy as np
from typing import Union, Type, Optional, List, Any
import numpy.typing as npt

def features_h5(
    features: npt.ArrayLike,
    labels: npt.ArrayLike,
    path: str
) -> None:
    """Write features and feature labels to an HDF5 file.

    This function creates an HDF5 file containing:
      - a ``features`` dataset (2D array),
      - a ``labels`` dataset storing variable-length UTF-8 strings.

    Parameters
    ----------
    features : ArrayLike
        2D array of shape ``(n_frames, n_features)`` containing feature values.
        The array is written directly with no dtype conversion.
    labels : ArrayLike
        A list/array of strings, one per feature column in ``features``.
    path : str
        Output path of the HDF5 file to be written.

    Returns
    -------
    None

    Raises
    ------
    OSError
        If the file cannot be created or written.
    TypeError
        If ``labels`` contains non-string elements that cannot be encoded.
    """
    hf = h5py.File(path, "w")
    hf.create_dataset("features", data=features)
    str_dtype = h5py.special_dtype(vlen=str)
    hf.create_dataset("labels", data=labels, dtype=str_dtype)
    hf.close()
    return

def pose_h5(
    pose: npt.ArrayLike,
    ids: npt.ArrayLike,
    path: str
) -> None:
    """Write pose data and per-frame IDs to an HDF5 file.

    This function writes two datasets:
      - ``pose``: a 3D array of shape ``(n_frames, n_keypoints, 3)``,
      - ``ids``: a vector of frame-level identifiers (e.g., video id).

    Parameters
    ----------
    pose : ArrayLike
        Array of 3D coordinates for each keypoint in each frame.
        Expected shape ``(n_frames, n_keypoints, 3)``.
    ids : ArrayLike
        Per-frame identifier array of length ``n_frames``.
    path : str
        Output path for the created `.h5` file.

    Returns
    -------
    None

    Raises
    ------
    OSError
        If the file cannot be opened or written.
    ValueError
        If ``pose`` and ``ids`` have incompatible lengths.
    """
    hf = h5py.File(path, "w")
    hf.create_dataset("pose", data=pose)
    hf.create_dataset("ids", data=ids)
    hf.close()
    return
