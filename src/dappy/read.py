import yaml
import h5py
import hdf5storage
from typing import Optional, Union, List, Tuple, Type
import pandas as pd
import numpy as np
from dappy.DataStruct import Connectivity
from tqdm import tqdm


def config(path: str):
    """
    Read configuration file and set instance attributes
    based on key, value pairs in the config file

    IN:
        filepath - Path to configuration file
    OUT:
        config_dict - Dict of path variables to data in config file
    """

    with open(path) as f:
        config_dict = yaml.safe_load(f)

    return config_dict


def meta(path, id: List[Union[str, int]]):
    meta = pd.read_csv(path)
    meta_by_frame = meta.iloc[id].reset_index().rename(columns={"index": "id"})
    meta = meta.reset_index().rename(columns={"index": "id"})

    return meta, meta_by_frame


def features_mat(
    analysis_path: Optional[str] = None,
    pose_path: Optional[str] = None,
    exp_key: Optional[str] = None,
    downsample: int = 20,
):
    """
    Load in data (we only care about id, frames_with_good_tracking and jt_features)

    IN:
        analysis_path - Path to MATLAB analysis struct with jt_features included
        pose_path - Path to predictions .mat file
        exp_key - Name of category to separate by experiment
        downsample - Factor by which to downsample features and IDs for analysis

    OUT:
        features - Numpy array of features for each frame for analysis (frames x features)
        id - List of labels for categories based on the exp_key
        frames_with_good_tracking - Indices in merged predictions file to keep track of downsampling
    """

    analysisstruct = hdf5storage.loadmat(
        analysis_path,
        variable_names=["jt_features", "frames_with_good_tracking", "tsnegranularity"],
    )
    features = analysisstruct["jt_features"].astype(np.float32)

    try:
        frames_with_good_tracking = (
            np.squeeze(analysisstruct["frames_with_good_tracking"][0][0].astype(int))
            - 1
        )
    except:
        frames_with_good_tracking = (
            np.squeeze(analysisstruct["frames_with_good_tracking"][0][1].astype(int))
            - 1
        )

    ids_full = np.squeeze(
        hdf5storage.loadmat(pose_path, variable_names=[exp_key])[exp_key].astype(int)
    )

    if np.min(ids_full) != 0:
        ids_full -= np.min(ids_full)

    id = ids_full[frames_with_good_tracking]  # Indexing out batch IDs

    print("Size of dataset: ", np.shape(features))

    # downsample
    frames_with_good_tracking = frames_with_good_tracking[::downsample]
    features = features[::downsample]
    id = id[::downsample]

    downsample = downsample * int(analysisstruct["tsnegranularity"])

    return features, id, frames_with_good_tracking


def pose_mat(
    path: str,
    connectivity: Connectivity,
    dtype: Optional[Type[Union[np.float64, np.float32]]] = np.float32,
):
    """Reads 3D pose data from .mat files.


    Parameters
    ----------
    path : str
        Path to pose `.mat` file.
    connectivity : Connectivity
        Connectivity object containing keypoint/joint/skeleta information.
    dtype : Optional[Type[Union[np.float64, np.float32]]], optional
        , by default np.float32

    Returns
    -------
    _type_
        _description_
    """

    try:
        f = h5py.File(path)["predictions"]
        mat_v7 = True
        total_frames = max(np.shape(f[list(f.keys())[0]]))
    except:
        print("Detected older version of '.mat' file")
        f = hdf5storage.loadmat(path, variable_names=["predictions"])["predictions"]
        mat_v7 = False
        total_frames = max(np.shape(f[0][0][0]))

    pose = np.empty((total_frames, 0, 3), dtype=dtype)
    for key in connectivity.joint_names:
        print(key)
        try:
            if mat_v7:
                joint_preds = np.expand_dims(np.array(f[key], dtype=dtype).T, axis=1)
            else:
                joint_preds = np.expand_dims(f[key][0][0].astype(dtype), axis=1)
        except:
            print("Could not find ", key, " in preds")
            continue

        pose = np.append(pose, joint_preds, axis=1)

    return pose


def ids(path: str, key: str):
    ids = np.squeeze(hdf5storage.loadmat(path, variable_names=[key])[key].astype(int))

    if np.min(ids) != 0:
        ids -= np.min(ids)
    return ids


def connectivity(path: str, skeleton_name: str):
    """_summary_

    Parameters
    ----------
    path : str
        Path to skeleton/connectivity Python file.
    skeleton_name : str
        Name of skeleton type to load in.

    Returns
    -------
    connectivity: Connectivity object
        Connectivity class object containing designated skeleton information
    """
    if path.endswith(".py"):
        import importlib.util

        mod_spec = importlib.util.spec_from_file_location("connectivity", path)
        con = importlib.util.module_from_spec(mod_spec)
        mod_spec.loader.exec_module(con)

        joint_names = con.JOINT_NAME_DICT[skeleton_name]  # joint names
        colors = con.COLOR_DICT[skeleton_name]  # color to be plotted for each linkage
        links = con.CONNECTIVITY_DICT[skeleton_name]  # joint linkages
        angles = con.JOINT_ANGLES_DICT[skeleton_name]  # angles to calculate

    connectivity = Connectivity(
        joint_names=joint_names, colors=colors, links=links, angles=angles
    )

    return connectivity


def features_h5(
    path, dtype: Optional[Type[Union[np.float64, np.float32]]] = np.float32
):
    """Reads feature array from an `.h5` file.

    Parameters
    ----------
    path : str
        Path to file.
    dtype : Optional[Type[Union[np.float64, np.float32]]], optional
        Desired data type of feature array. Can only be `np.float64` or `np.float32`, by default `np.float32`

    Returns
    -------
    features: np.ndarray
        2D array of features (# frames x # features).
    labels: List[str]
        List of labels for each column of features.
    """
    hf = h5py.File(path, "r")
    features = np.array(hf.get("features"), dtype=dtype)
    labels = np.array(hf.get("labels"), dtype=str).tolist()
    hf.close()
    print("Features loaded at path " + path)
    return features, labels


def pose_h5(
    path: str,
    dtype: Optional[Type[Union[np.float64, np.float32]]] = np.float32,
):
    """Reads 3D poses from an `.h5` file.

    Parameters
    ----------
    path : str
        Path to file.
    dtype : Optional[Type[Union[np.float64, np.float32]]], optional
        Desired data type of pose array. Can only be `np.float64` or `np.float32`, by default `np.float32`

    Returns
    -------
    pose : np.ndarray
        NumPy array of 3D pose values of shape (# frames x # joints x 3 coordinates).
    id : np.ndarray
        Id label for each frame in pose, e.g. video id.
    """
    hf = h5py.File(path, "r")
    pose = np.array(hf.get("pose"), dtype=dtype)
    id = np.array(hf.get("id"), dtype=np.int16)
    hf.close()
    return pose, id


def features_extended_h5(
    path: str,
    meta_dtype: Optional[Type] = str,
    dtype: Optional[Type[Union[np.float64, np.float32]]] = np.float32,
):
    hf = h5py.File(path, "r")
    features = np.array(hf.get("features"), dtype=dtype)
    labels = np.array(hf.get("labels"), dtype=str).tolist()
    id = np.array(hf.get("id"), dtype=np.int16)
    meta = np.array(hf.get("meta"), dtype=meta_dtype).tolist()
    clusters = np.array(hf.get("clusters"), dtype=np.int16)
    hf.close()
    print("Extended features loaded at path " + path)
    return features, labels, id, meta, clusters


def heuristics(path: str):
    import importlib.util

    mod_spec = importlib.util.spec_from_file_location("heuristics", path)
    heur = importlib.util.module_from_spec(mod_spec)
    mod_spec.loader.exec_module(heur)
    return heur


def pose_from_meta(
    path: str,
    connectivity: Connectivity,
    key: Optional[str] = "ClusterDirectory",
    dtype: Optional[Type[Union[np.float64, np.float32]]] = np.float32,
):
    """
    IN:
        path: path to metadata.csv file
    """
    meta = pd.read_csv(path)
    merged_pose = np.empty((0, len(connectivity.joint_names), 3), dtype=dtype)
    id = np.empty((0))
    for i, row in tqdm(meta.iterrows()):
        pose_path = row[key]
        meta_pose = pose_mat(pose_path, connectivity, dtype=dtype)
        merged_pose = np.append(merged_pose, meta_pose, axis=0)
        id = np.append(id, i * np.ones((meta_pose.shape[0])))

    meta_by_frame = meta.iloc[id].reset_index().rename(columns={"index": "id"})
    meta = meta.reset_index().rename(columns={"index": "id"})

    return merged_pose, id, meta, meta_by_frame
