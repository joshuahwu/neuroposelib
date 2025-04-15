import yaml
import h5py
import hdf5storage
from typing import Optional, Union, List, Tuple, Type, Dict
import pandas as pd
import numpy as np
from neuroposelib.DataStruct import Connectivity
from tqdm import tqdm
from scipy.io import loadmat as scipyloadmat
import numpy as np
import numpy.typing as npt


def config(path: str) -> dict:
    """Read configuration `.yaml` file and set instance attributes
    based on key, value pairs in the config file.

    Parameters
    ----------
    path : str
        Path to file.

    Returns
    -------
    config: dict
        Parameters from configuration file.
    """
    with open(path) as f:
        config = yaml.safe_load(f)

    return config


def meta(path: str, ids: List[Union[str, int]]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read in metadata `.csv` file.

    Parameters
    ----------
    path : str
        Path to file.
    ids : List[Union[str, int]]
        Id label for each frame in pose, e.g. video id (# frames).

    Returns
    -------
    meta : pd.DataFrame
        Metadata for each id, e.g. animal identity, treatment, sex (# ids, # metadata).
    meta_by_frame : pd.DataFrame
        Metadata for each frame (# frames, # metadata).
    """
    meta = pd.read_csv(path)
    meta_by_frame = meta.iloc[ids].reset_index().rename(columns={"index": "ids"})
    meta = meta.reset_index().rename(columns={"index": "ids"})

    return meta, meta_by_frame


def _features_mat(
    analysis_path: Optional[str] = None,
    pose_path: Optional[str] = None,
    exp_key: Optional[str] = None,
    downsample: int = 20,
):
    """DEPRECATING
    -----------
    Load in data outputs from CAPTURE

    Marshall, Jesse D., et al. "Continuous whole-body 3D kinematic recordings
    across the rodent behavioral repertoire." Neuron 109.3 (2021): 420-437.

    (we only care about ids, frames_with_good_tracking and jt_features)

    IN:
        analysis_path - Path to MATLAB analysis struct with jt_features included
        pose_path - Path to predictions `.mat` file
        exp_key - Name of category to separate by experiment
        downsample - Factor by which to downsample features and IDs for analysis

    OUT:
        features - Numpy array of features for each frame for analysis (# frames, # features)
        ids - List of labels for categories based on the exp_key
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

    ids = ids_full[frames_with_good_tracking]  # Indexing out batch IDs

    print("Size of dataset: ", np.shape(features))

    # downsample
    frames_with_good_tracking = frames_with_good_tracking[::downsample]
    features = features[::downsample]
    ids = ids[::downsample]

    downsample = downsample * int(analysisstruct["tsnegranularity"])

    return features, ids, frames_with_good_tracking


def pose_mat(
    path: str,
    connectivity: Connectivity,
    dtype: Optional[npt.DTypeLike] = np.float32,
) -> npt.NDArray:
    """Read pose array from `.mat` file.

    Parameters
    ----------
    path : str
        Path to file.
    connectivity : Connectivity
        Connectivity object containing keypoint/joint/skeletal information.
    dtype : Optional[npt.DTypeLike], optional
        Desired data type of output array.

    Returns
    -------
    pose : npt.NDArray
        Array of 3D pose values of shape (# frames, # keypoints, 3 coordinates).
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


def ids(path: str, key: str) -> npt.NDArray:
    """Read in ids from `.mat` pose files.

    Parameters
    ----------
    path : str
        Path to file.
    key : str
        Key in `.mat` file for ids.

    Returns
    -------
    ids : npt.NDArray
        Id label for each frame in pose, e.g. video id (# frames).
    """
    ids = np.squeeze(hdf5storage.loadmat(path, variable_names=[key])[key].astype(int))

    if np.min(ids) != 0:
        ids -= np.min(ids)
    return ids


def connectivity(path: str, skeleton_name: str) -> Connectivity:
    """DEPRECATING

    Reads in connectivity from skeleton.py file.

    Parameters
    ----------
    path : str
        Path to file.
    skeleton_name : str
        Name of skeleton type to load in.

    Returns
    -------
    connectivity: Connectivity
        Connectivity object containing keypoint/joint/skeletal information.
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


def connectivity_config(path: str) -> Connectivity:
    """Read in skeleton connectivity from skeleton config `.yaml` file.

    Parameters
    ----------
    path : str
        Path to file.

    Returns
    -------
    Connectivity
        Connectivity object containing keypoint/joint/skeletal information.
    """
    skeleton_config = config(path)

    joint_names = skeleton_config["LABELS"]
    colors = skeleton_config["COLORS"]
    links = skeleton_config["SEGMENTS"]
    keypt_colors = skeleton_config["KEYPT_COLORS"]

    if "JOINT_ANGLES" in skeleton_config.keys():
        angles = skeleton_config["JOINT_ANGLES"]
    else:
        angles = None

    connectivity = Connectivity(
        joint_names=joint_names,
        colors=colors,
        links=links,
        angles=angles,
        keypt_colors=keypt_colors,
    )

    return connectivity


def features_h5(
    path: str, dtype: Optional[npt.DTypeLike] = np.float32
) -> tuple[npt.NDArray, List[str]]:
    """Reads feature array from an `.h5` file.

    Parameters
    ----------
    path : str
        Path to file.
    dtype : Optional[npt.DTypeLike], optional
        Desired data type of output array.

    Returns
    -------
    features : npt.NDArray
        2D array of features (# frames, # features).
    labels : List[str]
        Labels for features in columns of features array.
    """
    hf = h5py.File(path, "r")
    features = np.array(hf.get("features"), dtype=dtype)
    labels = np.array(hf.get("labels"), dtype=str).tolist()
    hf.close()
    print("Features loaded at path " + path)
    return features, labels


def pose_h5(
    path: str,
    dtype: Optional[npt.DTypeLike] = np.float32,
) -> tuple[npt.NDArray, npt.NDArray]:
    """Reads 3D poses from an `.h5` file.

    Parameters
    ----------
    path : str
        Path to file.
    dtype : Optional[npt.DTypeLike], optional
        Desired data type of output array.

    Returns
    -------
    pose : npt.NDArray
        Array of 3D pose values of shape (# frames, # keypoints, 3 coordinates).
    ids : npt.NDArray
        Id label for each frame in pose, e.g. video id (# frames).
    """
    hf = h5py.File(path, "r")
    pose = np.array(hf.get("pose"), dtype=dtype)
    if "ids" in hf.keys():
        ids = np.array(hf.get("ids"), dtype=np.int16)
        hf.close()
        return pose, ids
    elif "id" in hf.keys():
        ids = np.array(hf.get("id"), dtype=np.int16)
        hf.close()
        return pose, ids
    else:
        hf.close()
        return pose


def _features_extended_h5(
    path: str,
    meta_dtype: Optional[Type] = str,
    dtype: Optional[npt.DTypeLike] = np.float32,
):
    hf = h5py.File(path, "r")
    features = np.array(hf.get("features"), dtype=dtype)
    labels = np.array(hf.get("labels"), dtype=str).tolist()
    ids = np.array(hf.get("ids"), dtype=np.int16)
    meta = np.array(hf.get("meta"), dtype=meta_dtype).tolist()
    clusters = np.array(hf.get("clusters"), dtype=np.int16)
    hf.close()
    print("Extended features loaded at path " + path)
    return features, labels, ids, meta, clusters


def _heuristics(path: str):
    import importlib.util

    mod_spec = importlib.util.spec_from_file_location("heuristics", path)
    heur = importlib.util.module_from_spec(mod_spec)
    mod_spec.loader.exec_module(heur)
    return heur


def pose_from_meta(
    path: str,
    connectivity: Connectivity,
    key: Optional[str] = "ClusterDirectory",
    file_type: Optional[str] = "dannce",
    dtype: Optional[npt.DTypeLike] = np.float32,
) -> tuple[npt.NDArray, npt.NDArray, pd.DataFrame, pd.DataFrame]:
    """Read pose array from a metadata file in which there are paths to individual pose files.

    Parameters
    ----------
    path : str
        Path to file.
    connectivity : Connectivity
        Connectivity object containing keypoint/joint/skeletal information.
    key : Optional[str], optional
        Column label in metadata corresponding to individual pose file paths.
    file_type : Optional[str], optional
        Origin of file type.
    dtype : Optional[npt.DTypeLike], optional
        Desired data type of output array.

    Returns
    -------
    pose : npt.NDArray
        Array of 3D pose values of shape (# frames, # keypoints, 3 coordinates).
    ids : npt.NDArray
        Id label for each frame in pose, e.g. video id (# frames).
    meta : pd.DataFrame
        Metadata for each id, e.g. animal identity, treatment, sex (# ids, # metadata).
    meta_by_frame : pd.DataFrame
        Metadata for each frame (# frames, # metadata).

    """
    meta = pd.read_csv(path)
    merged_pose = np.empty((0, len(connectivity.joint_names), 3), dtype=dtype)
    ids = np.empty((0))
    for i, row in tqdm(meta.iterrows()):
        pose_path = row[key]

        if file_type == "dannce":
            meta_pose = dannce_mat(pose_path, dtype=dtype)
        else:
            meta_pose = pose_mat(pose_path, connectivity, dtype=dtype)

        merged_pose = np.append(merged_pose, meta_pose, axis=0)
        ids = np.append(ids, i * np.ones((meta_pose.shape[0])))

    meta_by_frame = meta.iloc[ids].reset_index().rename(columns={"index": "ids"})
    meta = meta.reset_index().rename(columns={"index": "ids"})

    return merged_pose, ids, meta, meta_by_frame


def dannce_mat(
    path: str,
    dtype: Optional[npt.DTypeLike] = np.float32,
) -> npt.NDArray:
    """Read pose array from [DANNCE](https://github.com/spoonsso/dannce) output file.

    Parameters
    ----------
    path : str
        Path to file.
    dtype : Optional[npt.DTypeLike], optional
        Desired data type of output array.

    Returns
    -------
    pose : npt.NDArray
        Array of 3D pose values of shape (# frames, # keypoints, 3 coordinates).
    """
    mat_file = scipyloadmat(path, variable_names="pred")
    pose = np.moveaxis(mat_file["pred"], -1, -2).astype(dtype)

    return pose
