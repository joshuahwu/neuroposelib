import yaml
import h5py
import hdf5storage
from typing import Optional, Union, List, Tuple
import pandas as pd
import numpy as np
from DataStruct import Connectivity
from tqdm import tqdm


def config(path, config_params: Optional[List[str]] = None):
    """
    Read configuration file and set instance attributes
    based on key, value pairs in the config file

    IN:
        filepath - Path to configuration file
    OUT:
        config_dict - Dict of path variables to data in config file
    """
    # if config_params == 'paths_config':
    #     config_params = ['feature_path','pose_path','meta_path','out_path',
    #                      'skeleton_path','skeleton_name',
    #                      'exp_key']
    # elif config_params == 'params_config':
    #     config_params =

    # with open(filepath) as f:
    #     config_dict = yaml.safe_load(f)

    # for key in config_params:
    #     if key not in config_dict:
    #         config_dict[key]=None

    # return config_dict

    with open(path) as f:
        config_dict = yaml.safe_load(f)

    return config_dict


def meta(path, id: Optional[List[Union[str, int]]] = None):

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
    features = analysisstruct["jt_features"]

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


def pose(path: str, connectivity):

    try:
        f = h5py.File(path)["predictions"]
        mat_v7 = True
        total_frames = max(np.shape(f[list(f.keys())[0]]))
    except:
        print("Detected older version of '.mat' file")
        f = hdf5storage.loadmat(path, variable_names=["predictions"])["predictions"]
        mat_v7 = False
        total_frames = max(np.shape(f[0][0][0]))

    pose_3d = np.empty((total_frames, 0, 3))
    for key in connectivity.joint_names:
        print(key)
        try:
            if mat_v7:
                joint_preds = np.expand_dims(np.array(f[key]).T, axis=1)
            else:
                joint_preds = np.expand_dims(f[key][0][0], axis=1)
        except:
            print("Could not find ", key, " in preds")
            continue

        pose_3d = np.append(pose_3d, joint_preds, axis=1)

    return pose_3d

def ids(path, key):
    ids = np.squeeze(
            hdf5storage.loadmat(path, variable_names=[key])[key].astype(int)
        )

    if np.min(ids) != 0:
        ids -= np.min(ids)
    return ids


def connectivity(path: str, skeleton_name: str):

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


def features_h5(path):
    """
    Reads h5 file for features and labels
    """
    hf = h5py.File(path, "r")
    features = np.array(hf.get("features"))
    labels = np.array(hf.get("labels"), dtype=str).tolist()
    hf.close()
    print("Features loaded at path " + path)
    return features, labels


def pose_h5(path, exp_key):
    hf = h5py.File(path, "r")
    pose = np.array(hf.get("pose"))
    id = np.array(hf.get(exp_key))
    hf.close()
    return pose, id


def heuristics(path):
    import importlib.util

    mod_spec = importlib.util.spec_from_file_location("heuristics", path)
    heur = importlib.util.module_from_spec(mod_spec)
    mod_spec.loader.exec_module(heur)
    return heur

def pose_from_meta(path, connectivity):
    """
    IN:
        path: path to metadata.csv file
    """
    meta = pd.read_csv(path)
    merged_pose = np.empty((0, len(connectivity.joint_names), 3))
    id = np.empty((0))
    for i, row in tqdm(meta.iterrows()):
        pose_path = row["ClusterDirectory"]
        meta_pose = pose(pose_path, connectivity, exp_key=None)
        merged_pose = np.append(merged_pose, meta_pose, axis=0)
        id = np.append(id, i * np.ones((meta_pose.shape[0])))

    meta_by_frame = meta.iloc[id].reset_index().rename(columns={"index": "id"})
    meta = meta.reset_index().rename(columns={"index": "id"})

    return merged_pose, id, meta, meta_by_frame
