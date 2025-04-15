import h5py
import numpy as np
from typing import Union, Type, Optional, List
import numpy.typing as npt


def features_h5(features: npt.ArrayLike, labels: npt.ArrayLike, path: str):
    """Writes features and labels to `.h5` file.

    Parameters
    ----------
    features : npt.ArrayLike
        2D array of features (# frames, # features).
    labels : npt.ArrayLike
        List of labels for features in columns of features array.
    path : str
        Path to file.
    """

    hf = h5py.File(path, "w")
    hf.create_dataset("features", data=features)
    str_dtype = h5py.special_dtype(vlen=str)
    hf.create_dataset("labels", data=labels, dtype=str_dtype)
    hf.close()
    return


def pose_h5(pose: npt.ArrayLike, ids: npt.ArrayLike, path: str):
    """Writes poses to `.h5` file.

    Parameters
    ----------
    pose : npt.ArrayLike
        Array of 3D pose values of shape (# frames, # keypoints, 3 coordinates).
    ids : npt.ArrayLike
        Id label for each frame in pose, e.g. video id (# frames).
    path : str
        Path to file.
    """
    hf = h5py.File(path, "w")
    hf.create_dataset("pose", data=pose)
    hf.create_dataset("ids", data=ids)
    hf.close()
    return


# def extended_features_h5(
#     features: npt.ArrayLike,
#     labels: List[str],
#     ids: npt.ArrayLike,
#     meta: npt.ArrayLike,
#     clusters: npt.ArrayLike,
#     path: str,
# ):
#     """
#     DEPRECATING
#     Write extended set of features, metadata, and clusters to `.h5` file.

#     Parameters
#     ----------
#     features : npt.ArrayLike
#         Feature array (# frames, # features)
#     labels : List[str]
#         _description_
#     ids : npt.ArrayLike
#         _description_
#     meta : npt.ArrayLike
#         _description_
#     clusters : npt.ArrayLike
#         _description_
#     path : str
#         _description_
#     """
#     hf = h5py.File(path, "w")
#     hf.create_dataset("features", data=features)
#     hf.create_dataset("labels", data=labels)
#     hf.create_dataset("ids", data=ids)
#     hf.create_dataset("clusters", data=clusters)

#     if isinstance(meta[0], str):
#         str_dtype = h5py.special_dtype(vlen=str)
#         hf.create_dataset("meta", data=meta, dtype=str_dtype)
#     else:
#         hf.create_dataset("meta", data=meta)
#     return
