import pandas as pd
import numpy as np
from typing import Optional, Union, List, Tuple, Type
from pathlib import Path
import numpy.typing as npt

class Connectivity:
    """
    Class for storing keypoint and linkage settings for 3D pose estimation skeletons
    """

    def __init__(
        self,
        joint_names: List[str],
        colors: Union[npt.ArrayLike, List[Tuple[float, float, float, float]]],
        links: Union[npt.ArrayLike, List[Tuple[int, int]]],
        angles: Optional[Union[npt.ArrayLike, List[Tuple[int, int, int]]]] = None,
        keypt_colors: Optional[
            Union[npt.ArrayLike, List[Tuple[float, float, float, float]]]
        ] = None,
    ):
        """Initializes instance of Connectivity class

        Parameters
        ----------
        joint_names : List[str]
            List of names of all joints/keypoints in skeleton.
        colors : Union[npt.ArrayLike, List[Tuple[float, float, float, float]]]
            RGB+A color values by which to plot each linkage.
        links : Union[npt.ArrayLike, List[Tuple[int, int]]]
            Identifies skeletal links between joints/keypoints by index within joint_names
        angles : Union[npt.ArrayLike, List[Tuple[int, int, int]]]
            Designations of 3 joints between which a vector angle is formed. Middle value indicates
             common point from which 2 vectors are formed.
        """

        self.joint_names = joint_names
        self.colors = self._check_type(colors, np.float32)
        self.links = self._check_type(links, np.uint16)
        if keypt_colors is not None:
            self.keypt_colors = self._check_type(keypt_colors, np.float32)
        if angles is not None:
            self.angles = self._check_type(angles, np.uint16)

    def _check_type(
        self,
        in_arr: Union[npt.ArrayLike, List[Tuple]],
        dtype: npt.DTypeLike,
    ) -> npt.ArrayLike:
        """Checks the type of input and converts to NumPy array of desired data type.

        Parameters
        ----------
        in_arr : Union[npt.ArrayLike, List[Tuple]]
            Input to convert.
        dtype : npt.DTypeLike
            Data type to which input should be converted.

        Returns
        -------
        npt.ArrayLike(dtype=dtype)
            NumPy array with specified data type.
        """
        if isinstance(in_arr, list):
            return np.array(in_arr, dtype=dtype)
        elif in_arr.dtype != dtype:
            return in_arr.astype(dtype)
        else:
            return in_arr

class DataStruct:
    """DEPRECATING
    Class for organizing and linking metadata to features and pose data
    """

    # TODO: Refactor this class/potentially deprecate
    # TODO: If refactor, make categorical meta fields to be sparse matrices.
    # TODO: Another idea is to use this to store analysis transform objects
    # (e.g. pca, umap, t-sne, watershed)

    _props = [
        "embed_vals",
        "data",
        "meta_by_frame",
        "frame",
        "features",
        "meta",
    ]

    def __init__(
        self,
        data: pd.DataFrame = pd.DataFrame(),
        id: Optional[Union[List, npt.ArrayLike]] = None,
        meta: pd.DataFrame = pd.DataFrame(),
        meta_by_frame: pd.DataFrame = pd.DataFrame(),
        pose: npt.ArrayLike = None,
        connectivity=None,
        frame: Optional[Union[List, npt.ArrayLike]] = None,
        feature_labels: Optional[List[str]] = None,
    ):
        self.data = data
        self.meta = meta
        self.connectivity = connectivity

        if (id is not None) or ("id" not in self.data):
            self.id = id

        self.pose = pose
        self.feature_labels = feature_labels

        if frame is not None:
            self.frame = frame
        elif "frame" not in self.data:
            self.frame = np.arange(0, self.data.shape[0])

        self.meta_by_frame = meta_by_frame
        # import pdb; pdb.set_trace()

    def __getitem__(self, idx):
        if not isinstance(idx, tuple):
            idx = tuple(idx)
        new_struct = DataStruct(
            data=self.data.loc[idx].reset_index(drop=True),
            pose=self.pose,
            meta=self.meta,
            feature_labels=self.feature_labels,
            connectivity=self.connectivity,
        )
        return new_struct

    def check_reset_data(self, len: int):
        if self.data.shape[0] != len:
            self.data = pd.DataFrame()

    @property
    def frame(self):
        return self.data["frame"].to_numpy()

    @frame.setter
    def frame(self, frame: Union[List[int], np.array]):
        self.data["frame"] = frame

    @property
    def id(self):
        return self.data["id"].to_numpy()

    @id.setter
    def id(self, id: Union[List[Union[str, int]], np.array]):
        self.data["id"] = id

    @property
    def meta_by_frame(self):
        return self.data[list(self.meta.columns.values)]

    @meta_by_frame.setter
    def meta_by_frame(self, meta_by_frame: pd.DataFrame):
        self.data.loc[:, list(meta_by_frame.columns.values)] = meta_by_frame.values

    def meta_unique(self, column_id: str):
        return list(set(list(self.data[column_id].values)))

    @property
    def n_frames(self):
        return self.data.shape[0]

    @property
    def embed_vals(self):
        return np.array(list(self.data["embed_vals"].to_numpy()))

    @embed_vals.setter
    def embed_vals(self, embed_vals: Optional[np.array] = None):
        self.data["embed_vals"] = list(embed_vals)

    @property
    def features(self):
        return np.array(list(self.data["features"].to_numpy()))  # seems slow

    @features.setter
    def features(self, features: np.array):
        self.data["features"] = list(features)  # seems slow

    @property
    def feat_shape(self):
        return np.shape(self.features)

    def write_pickle(self, out_path: Optional[str] = None):
        # if out_path is None:
        #     out_path = self.out_path
        import pickle

        Path(out_path).mkdir(parents=True, exist_ok=True)

        pickle.dump(self, open("".join([out_path, "datastruct.p"]), "wb"))

    def downsample(self, downsample: int):
        """
        TODO: Separately downsamples by id
        """
        return self[::downsample]

    def get_frequencies(self, cat1="id", cat2="Cluster"):
        """
        Calculates the percentage of time each cat1 spends in each cat2
        OUT:
            freq: numpy array of (# videos x # clusters)
        """
        num_clusters = np.max(self.data[cat2]) + 1
        freq = np.zeros((len(self.meta[cat1]), num_clusters))
        for i in self.meta[cat1]:
            cluster_by_exp = self.data[cat2][self.data[cat1] == i]
            freq[i, :] = np.histogram(
                cluster_by_exp, bins=num_clusters, range=(-0.5, num_clusters - 0.5)
            )[0]
        frame_totals = np.sum(freq, axis=1)
        frame_totals = np.where(frame_totals == 0, 1, frame_totals)
        freq = freq / np.expand_dims(frame_totals, axis=1)

        return freq
