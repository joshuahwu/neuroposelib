import h5py
import numpy as np
from typing import Union, Type, Optional, List


def features_h5(features: np.ndarray, labels: List[str], path: str):
    """
    Writes features and labels to h5 file
    """
    hf = h5py.File(path, "w")
    hf.create_dataset("features", data=features)
    str_dtype = h5py.special_dtype(vlen=str)
    hf.create_dataset("labels", data=labels, dtype=str_dtype)
    hf.close()
    return


def pose_h5(pose: np.ndarray, id: Type[Union[np.ndarray, List]], path: str):
    """
    Writes poses (#frames x #joints x `xyz` to h5 file)
    """
    hf = h5py.File(path, "w")
    hf.create_dataset("pose", data=pose)
    hf.create_dataset("id", data=id)
    hf.close()
    return


def extended_features_h5(
    features: np.ndarray,
    labels: List[str],
    id: Type[Union[np.ndarray, List]],
    meta: Type[Union[np.ndarray, List]],
    clusters: Type[Union[np.ndarray, List]],
    path: str,
):
    hf = h5py.File(path, "w")
    hf.create_dataset("features", data=features)
    hf.create_dataset("labels", data=labels)
    hf.create_dataset("id", data=id)
    hf.create_dataset("clusters", data=clusters)

    if isinstance(meta[0], str):
        str_dtype = h5py.special_dtype(vlen=str)
        hf.create_dataset("meta", data=meta, dtype=str_dtype)
    else:
        hf.create_dataset("meta", data=meta)
    return
