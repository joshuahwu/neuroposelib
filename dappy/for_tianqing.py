import h5py
from typing import Optional, Union, List, Tuple, Type
import numpy as np
import pickle

def read_pose_h5(path:str, dtype: Optional[Type[Union[np.float64, np.float32]]] = np.float32,):
    """ Reads 3D poses from an `.h5` file.

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
    pose = np.array(hf.get("pose"),dtype=dtype)
    id = np.array(hf.get("id"),dtype=np.int16)
    hf.close()
    return pose, id


# Read aligned pose (not spine centered)
pose,vid_id = read_pose_h5('/home/exx/Desktop/GitHub/CAPTURE_data/ensemble_healthy/pose_aligned.h5')


## I downsampled 
data_obj = pickle.load(
    open("/home/exx/Desktop/GitHub/results/ensemble_healthy/fitsne/datastruct.p", "rb")
)

cluster_ids_by_frame = data_obj.data["Cluster"].values
import pdb; pdb.set_trace()