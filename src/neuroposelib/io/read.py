import yaml
import h5py
import hdf5storage
from typing import Optional, Union, List, Tuple, Type, Dict
import pandas as pd
import numpy as np
from neuroposelib.DataStruct import Connectivity
from tqdm import tqdm
from scipy.io import loadmat as scipyloadmat
import numpy.typing as npt

def cluster_annotations(filepath: str) -> pd.DataFrame:
    """Load sparse cluster→annotation CSV and return per-cluster annotation names.

    This CSV format is expected to be a *sparse annotation matrix*:

    - Rows are indexed by a ``Cluster`` id.
    - Columns are annotation labels (e.g. ``grooming``, ``rearing``).
    - Cells contain a marker (e.g., ``1`` or ``x``) to indicate that
      the annotation applies to that cluster. Unmarked entries are empty/NaN.

    The function returns a tidy DataFrame listing, for each cluster,
    the **single** annotation that applies.  
    If any row has **more than one** marked annotation, an error is raised.

    Parameters
    ----------
    filepath : str
        Path to the CSV file. Must contain a ``Cluster`` column.

    Returns
    -------
    pandas.DataFrame
        A DataFrame indexed by ``cluster`` with a single column
        ``annotations`` specifying the annotation label.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    pandas.errors.EmptyDataError
        If the CSV is empty.
    KeyError
        If the CSV does not contain a ``Cluster`` column.
    ValueError
        If any cluster row has more than one non-empty annotation entry.

    Examples
    --------
    >>> # CSV example:
    >>> # Cluster, grooming, rearing, walking
    >>> # 0, 1, , 
    >>> # 1, , x,
    >>> df = cluster_annotations("annotations.csv")
    >>> df
    # cluster | annotations
    # 0       | grooming
    # 1       | rearing
    """
    # Load CSV with Cluster as index
    annotations = pd.read_csv(filepath, index_col="Cluster")

    # Count non-empty markers per row
    # Anything that is not NaN or empty string counts as a marker
    marker_mask = annotations.notna() & (annotations.astype(str).str.strip() != "")
    marker_counts = marker_mask.sum(axis=1)

    # Raise if any row has > 1 marked annotation
    if (marker_counts > 1).any():
        bad_clusters = marker_counts[marker_counts > 1].index.tolist()
        raise ValueError(
            f"Cluster(s) {bad_clusters} contain more than one annotation marker. "
            "Each cluster must have exactly one annotation."
        )

    # Now stack, preserving only the annotation name (column name)
    stacked = annotations.stack()
    stacked.index = stacked.index.set_names(["cluster", "annotations"])
    out = stacked.reset_index(level=1).drop(columns=0)

    return out


def config(path: str) -> Dict:
    """Read YAML configuration file.

    The YAML file is parsed with :func:`yaml.safe_load` and returned as a
    Python dictionary.

    Parameters
    ----------
    path : str
        Path to the configuration YAML file.

    Returns
    -------
    dict
        Parsed key/value pairs from the configuration file.
    """
    with open(path) as f:
        config = yaml.safe_load(f)
    return config


def meta(path: str, ids: List[Union[str, int]]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Read metadata CSV and return per-id and per-frame tables.

    The function loads a CSV with :func:`pd.read_csv`, then creates a
    ``meta_by_frame`` DataFrame by indexing into the loaded table with
    ``ids`` (which is typically a list/array mapping frame→id).

    Parameters
    ----------
    path : str
        Path to CSV file.
    ids : list of (str or int)
        Labels per frame (e.g., video/experiment id for each frame).

    Returns
    -------
    meta : pandas.DataFrame
        Metadata for each id (rows correspond to unique ids in the CSV).
    meta_by_frame : pandas.DataFrame
        Metadata expanded to one row per frame (length equals number of frames).
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
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load features and ids from MATLAB analysis and predictions.

    DEPRECATION NOTE
    --------------
    This helper was written to load outputs from the CAPTURE analysis
    pipeline and the associated MATLAB prediction files.

    Behavior
    --------
    - Loads ``jt_features`` and ``frames_with_good_tracking`` from an analysis
      MATLAB struct (via [`hdf5storage`](https://pypi.org/project/hdf5storage/)).
    - Loads experiment ids (``exp_key``) from the predictions MATLAB file.
    - Optionally downsamples frames and features by ``downsample``.

    Parameters
    ----------
    analysis_path : str, optional
        Path to MATLAB analysis struct (must contain ``jt_features``).
    pose_path : str, optional
        Path to predictions `.mat` file (contains experiment ids).
    exp_key : str, optional
        Key to load experiment ids from the pose `.mat`.
    downsample : int, optional
        Factor by which to downsample frames/features (default: 20).

    Returns
    -------
    features : ndarray (float32)
        Feature matrix of shape ``(n_frames, n_features)`` (may be downsampled).
    ids : ndarray (int)
        Per-frame experiment ids (may be downsampled).
    frames_with_good_tracking : ndarray (int)
        Indices of (original) frames considered to have good tracking.
    """
    analysisstruct = hdf5storage.loadmat(
        analysis_path, variable_names=["jt_features", "frames_with_good_tracking", "tsnegranularity"]
    )
    features = analysisstruct["jt_features"].astype(np.float32)
    try:
        frames_with_good_tracking = (
            np.squeeze(analysisstruct["frames_with_good_tracking"][0][0].astype(int)) - 1
        )
    except Exception:
        frames_with_good_tracking = (
            np.squeeze(analysisstruct["frames_with_good_tracking"][0][1].astype(int)) - 1
        )

    ids_full = np.squeeze(hdf5storage.loadmat(pose_path, variable_names=[exp_key])[exp_key].astype(int))
    if np.min(ids_full) != 0:
        ids_full -= np.min(ids_full)
    ids = ids_full[frames_with_good_tracking]

    # Indexing out batch IDs
    print("Size of dataset: ", np.shape(features))

    # downsample
    frames_with_good_tracking = frames_with_good_tracking[::downsample]
    features = features[::downsample]
    ids = ids[::downsample]
    downsample = downsample * int(analysisstruct["tsnegranularity"])
    return features, ids, frames_with_good_tracking


def pose_mat(
    path: str, connectivity: Connectivity, dtype: Optional[npt.DTypeLike] = np.float32
) -> npt.NDArray[Any]:
    """Read pose array from a MATLAB `.mat` predictions file.

    Supports both HDF5-backed v7+ MATLAB files (accessed via :mod:`h5py`) and
    older formats read through :mod:`hdf5storage`.

    Parameters
    ----------
    path : str
        Path to the `.mat` file containing a ``predictions`` variable.
    connectivity : Connectivity
        Connectivity object containing ``joint_names`` (used to order joints).
    dtype : numpy dtype-like, optional
        Desired dtype for the returned array (default ``np.float32``).

    Returns
    -------
    pose : ndarray
        Pose array shaped ``(n_frames, n_keypoints, 3)``.
    """
    try:
        f = h5py.File(path)["predictions"]
        mat_v7 = True
        total_frames = max(np.shape(f[list(f.keys())[0]]))
    except Exception:
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
        except Exception:
            print("Could not find ", key, " in preds")
            continue
        pose = np.append(pose, joint_preds, axis=1)
    return pose


def ids(path: str, key: str) -> npt.NDArray[np.int_]:
    """Read per-frame ids from `.mat` files.

    Parameters
    ----------
    path : str
        Path to `.mat` file.
    key : str
        Variable name inside the `.mat` file that stores ids.

    Returns
    -------
    ids : ndarray (int)
        Array of per-frame ids (zero-indexed).
    """
    ids = np.squeeze(hdf5storage.loadmat(path, variable_names=[key])[key].astype(int))
    if np.min(ids) != 0:
        ids -= np.min(ids)
    return ids


def connectivity(path: str, skeleton_name: str) -> Connectivity:
    """(DEPRECATED) Load a Connectivity object from a Python skeleton definition.

    The function expects a Python file at ``path`` providing constants such as
    ``JOINT_NAME_DICT``, ``COLOR_DICT``, ``CONNECTIVITY_DICT`` and
    ``JOINT_ANGLES_DICT`` keyed by ``skeleton_name``. It imports the file as a
    module and constructs a :class:`Connectivity`.

    Parameters
    ----------
    path : str
        Path to a Python file that defines skeleton dictionaries.
    skeleton_name : str
        Name of the skeleton to load (a key in the file's dictionaries).

    Returns
    -------
    connectivity : Connectivity
        Constructed connectivity object.
    """
    if path.endswith(".py"):
        import importlib.util

        mod_spec = importlib.util.spec_from_file_location("connectivity", path)
        con = importlib.util.module_from_spec(mod_spec)
        mod_spec.loader.exec_module(con)

        joint_names = con.JOINT_NAME_DICT[skeleton_name]
        colors = con.COLOR_DICT[skeleton_name]
        links = con.CONNECTIVITY_DICT[skeleton_name]
        angles = con.JOINT_ANGLES_DICT[skeleton_name]
        connectivity_obj = Connectivity(
            joint_names=joint_names, colors=colors, links=links, angles=angles
        )
        return connectivity_obj
    # If not a .py file, fall back to returning the Connectivity type itself
    return Connectivity


def connectivity_config(path: str) -> Connectivity:
    """Load skeleton connectivity from a YAML skeleton config.

    Parameters
    ----------
    path : str
        Path to skeleton YAML configuration.

    Returns
    -------
    Connectivity
        Connectivity object constructed from the YAML keys.
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
    connectivity_obj = Connectivity(
        joint_names=joint_names,
        colors=colors,
        links=links,
        angles=angles,
        keypt_colors=keypt_colors,
    )
    return connectivity_obj


def features_h5(path: str, dtype: Optional[npt.DTypeLike] = np.float32) -> Tuple[npt.NDArray[Any], List[str]]:
    """Read features and labels from an HDF5 file.

    Parameters
    ----------
    path : str
        Path to `.h5` file containing ``features`` and ``labels`` datasets.
    dtype : numpy dtype-like, optional
        Desired dtype for the returned features (default ``np.float32``).

    Returns
    -------
    features : ndarray
        2D array of features (``n_frames x n_features``).
    labels : list of str
        Column labels for the features.
    """
    hf = h5py.File(path, "r")
    features = np.array(hf.get("features"), dtype=dtype)
    labels = np.array(hf.get("labels"), dtype=str).tolist()
    hf.close()
    print("Features loaded at path " + path)
    return features, labels


def pose_h5(path: str, dtype: Optional[npt.DTypeLike] = np.float32) -> Tuple[npt.NDArray[Any], Optional[npt.NDArray[np.int_]]]:
    """Read poses (and optional ids) from an HDF5 file.

    Parameters
    ----------
    path : str
        Path to `.h5` file containing a ``pose`` dataset and optionally
        ``ids`` or ``id``.
    dtype : numpy dtype-like, optional
        Desired dtype for pose (default ``np.float32``).

    Returns
    -------
    pose : ndarray
        Pose array of shape ``(n_frames, n_keypoints, 3)``.
    ids : ndarray or None
        If present, an integer array of per-frame ids (otherwise ``None``).
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
        return pose, None


def _features_extended_h5(
    path: str, meta_dtype: Optional[Type] = str, dtype: Optional[npt.DTypeLike] = np.float32
) -> Tuple[npt.NDArray[Any], List[str], npt.NDArray[np.int_], List, npt.NDArray[np.int_]]:
    """Read extended features and metadata from an HDF5 file.

    This helper returns features, labels, ids, meta and cluster assignments.

    Parameters
    ----------
    path : str
        Path to `.h5` file.
    meta_dtype : type, optional
        Data type for metadata (default: ``str``).
    dtype : numpy dtype-like, optional
        Desired dtype for features.

    Returns
    -------
    features, labels, ids, meta, clusters
    """
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
    """Dynamically import a heuristics Python file.

    Parameters
    ----------
    path : str
        Path to Python module implementing heuristics.

    Returns
    -------
    module
        Imported heuristics module.
    """
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
) -> Tuple[npt.NDArray[Any], npt.NDArray[np.int_], pd.DataFrame, pd.DataFrame]:
    """Construct a merged pose array from metadata listing individual pose files.

    The function reads a metadata CSV where one column points to individual
    pose files (``key``). For each row it reads the pose file (using
    :func:`dannce_mat` or :func:`pose_mat` depending on ``file_type``)
    and concatenates results into a single large pose array, returning also
    a per-frame ``ids`` vector and both ``meta`` tables.

    Parameters
    ----------
    path : str
        Path to metadata CSV.
    connectivity : Connectivity
        Connectivity object used for reading individual pose files.
    key : str, optional
        Column label in the metadata pointing to pose file paths.
    file_type : str, optional
        Origin file type; ``'dannce'`` will use :func:`dannce_mat`.
    dtype : numpy dtype-like, optional
        Desired dtype for the merged poses.

    Returns
    -------
    merged_pose : ndarray
        Merged pose array (``n_frames_total, n_keypoints, 3``).
    ids : ndarray (float)
        Per-frame ids (float array in original implementation).
    meta : pandas.DataFrame
        Per-id metadata table.
    meta_by_frame : pandas.DataFrame
        Per-frame expanded metadata table.
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


def dannce_mat(path: str, dtype: Optional[npt.DTypeLike] = np.float32) -> npt.NDArray[Any]:
    """Read pose from DANNCE MATLAB output.

    DANNCE stores predicted poses in a ``pred`` variable. This helper uses
    :func:`scipy.io.loadmat` (imported as ``scipyloadmat``) and reshapes the
    axis order to match the other loaders.

    Parameters
    ----------
    path : str
        Path to DANNCE `.mat` file.
    dtype : numpy dtype-like, optional
        Desired dtype of returned pose (default ``np.float32``).

    Returns
    -------
    pose : ndarray
        Pose array shaped ``(n_frames, n_keypoints, 3)``.
    """
    mat_file = scipyloadmat(path, variable_names="pred")
    pose = np.moveaxis(mat_file["pred"], -1, -2).astype(dtype)
    return pose
