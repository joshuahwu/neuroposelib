import pandas as pd
import numpy as np
from typing import Optional, Union, List, Tuple, Type, Any
from pathlib import Path
import numpy.typing as npt


class Connectivity:
    """
    Store keypoint/link settings for 3D-pose skeletons.

    Parameters
    ----------
    joint_names : list of str
        Names of all joints/keypoints in the skeleton.
    colors : ndarray or list of tuples, shape (n_links, 4)
        RGBA color per linkage (values 0..1).
    links : ndarray or list of tuples, shape (n_links, 2)
        Pairs of joint indices (ints) defining each link.
    angles : ndarray or list of tuples, optional, shape (n_angles, 3)
        Triplets of joint indices defining angles (middle index is the vertex).
    keypt_colors : ndarray or list of tuples, optional, shape (n_keypts, 4)
        RGBA color per keypoint.
    """

    def __init__(
        self,
        joint_names: List[str],
        colors: Union[npt.NDArray[Any], List[Tuple[float, float, float, float]]],
        links: Union[npt.NDArray[Any], List[Tuple[int, int]]],
        angles: Optional[Union[npt.NDArray[Any], List[Tuple[int, int, int]]]] = None,
        keypt_colors: Optional[
            Union[npt.NDArray[Any], List[Tuple[float, float, float, float]]]
        ] = None,
    ) -> None:
        self.joint_names: List[str] = joint_names
        # Store arrays with consistent dtypes
        self.colors: npt.NDArray[Any] = self._check_type(colors, np.float32)
        self.links: npt.NDArray[Any] = self._check_type(links, np.uint16)
        if keypt_colors is not None:
            self.keypt_colors: npt.NDArray[Any] = self._check_type(keypt_colors, np.float32)
        else:
            self.keypt_colors = None
        if angles is not None:
            self.angles: npt.NDArray[Any] = self._check_type(angles, np.uint16)
        else:
            self.angles = None

    def _check_type(
        self,
        in_arr: Union[npt.NDArray[Any], List[Tuple[Any, ...]]],
        dtype: npt.DTypeLike,
    ) -> npt.NDArray[Any]:
        """
        Ensure `in_arr` is a NumPy array with dtype `dtype`.

        Parameters
        ----------
        in_arr : ndarray or list of tuples
            Input array-like to convert.
        dtype : dtype-like
            Target NumPy dtype.

        Returns
        -------
        arr : ndarray, dtype=dtype
            Converted array.
        """
        if isinstance(in_arr, list):
            return np.array(in_arr, dtype=dtype)
        # Accept numpy arrays (or array-like with `.dtype`)
        arr = np.asarray(in_arr)
        if arr.dtype != dtype:
            return arr.astype(dtype)
        else:
            return arr



# TODO: Refactor this class/potentially deprecate
    # TODO: If refactor, make categorical meta fields to be sparse matrices.
    # TODO: Another idea is to use this to store analysis transform objects
    # (e.g. pca, umap, t-sne, watershed)
class DataStruct:
    """
    Container linking dataframe rows to pose/features/metadata.

    Notes
    -----
    This class is a thin wrapper around pandas DataFrame rows with convenience
    accessors for common fields (frame, id, features, embed_vals, etc.).
    """

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
        ids: Optional[Union[List[Any], npt.NDArray[Any]]] = None,
        meta: pd.DataFrame = pd.DataFrame(),
        meta_by_frame: pd.DataFrame = pd.DataFrame(),
        pose: Optional[npt.NDArray[Any]] = None,
        connectivity: Optional[Connectivity] = None,
        frame: Optional[Union[List[int], npt.NDArray[Any]]] = None,
        feature_labels: Optional[List[str]] = None,
    ) -> None:
        """
        Parameters
        ----------
        data : pandas.DataFrame, shape (n_rows, n_columns)
            Primary table storing per-frame information; columns may include 'frame', 'id', 'features', 'embed_vals', etc.
        ids : list or ndarray, optional, shape (n_videos,) or (n_rows,)
            Identifier(s) associated with rows or videos.
        meta : pandas.DataFrame, shape (n_videos, n_meta_cols)
            Per-video metadata table.
        meta_by_frame : pandas.DataFrame, shape (n_rows, n_meta_cols)
            Per-frame metadata to be appended to `data`.
        pose : ndarray, optional, shape (n_rows, n_keypoints, 3)
            Pose arrays aligned to data rows.
        connectivity : Connectivity or None
            Skeleton connectivity object.
        frame : list or ndarray, optional, shape (n_rows,)
            Frame indices for each row. If None and `data` lacks 'frame', generated automatically.
        feature_labels : list of str, optional
            Labels for columns/features in `features`.
        """
        self.data: pd.DataFrame = data.copy()
        self.meta: pd.DataFrame = meta.copy()
        self.connectivity: Optional[Connectivity] = connectivity

        # If an id argument is provided or 'id' missing in data, set it
        if (ids is not None) or ("ids" not in self.data.columns):
            self.ids = ids

        self.pose: Optional[npt.NDArray[Any]] = pose
        self.feature_labels: Optional[List[str]] = feature_labels

        if frame is not None:
            self.frame = frame
        elif "frame" not in self.data.columns:
            # default frame indices 0..n-1
            self.frame = np.arange(0, self.data.shape[0])

        # populate meta columns into data if provided
        self.meta_by_frame = meta_by_frame.copy()

    def __getitem__(self, idx: Any) -> "DataStruct":
        """
        Slice DataStruct by DataFrame-style indexing.

        Parameters
        ----------
        idx : indexer
            Indexing spec for pandas.DataFrame.loc (can be slice, list, boolean mask, etc.)

        Returns
        -------
        new_struct : DataStruct
            New DataStruct containing subsetted rows (data reindexed from 0).
        """
        # Use pandas indexing semantics by delegating to .loc when possible
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

    def check_reset_data(self, length: int) -> None:
        """
        Reset `self.data` to empty DataFrame if its number of rows differs from `length`.

        Parameters
        ----------
        length : int
            Expected number of rows; if mismatch, `self.data` is cleared.
        """
        if self.data.shape[0] != length:
            self.data = pd.DataFrame()

    @property
    def frame(self) -> npt.NDArray[Any]:
        """
        Return frame indices for each row.

        Returns
        -------
        frames : ndarray, shape (n_rows,)
            Frame indices (integer).
        """
        return self.data["frame"].to_numpy()

    @frame.setter
    def frame(self, frame: Union[List[int], npt.NDArray[Any]]) -> None:
        """
        Set the `frame` column in `self.data`.

        Parameters
        ----------
        frame : list or ndarray, shape (n_rows,)
            Frame indices to assign.
        """
        self.data["frame"] = frame

    @property
    def ids(self) -> npt.NDArray[Any]:
        """
        Return 'ids' column as ndarray.

        Returns
        -------
        ids : ndarray, shape (n_rows,) or (n_videos,)
        """
        return self.data["ids"].to_numpy()

    @ids.setter
    def ids(self, ids: Union[List[Union[str, int]], npt.NDArray[Any]]) -> None:
        """
        Set the `ids` column in `self.data`.

        Parameters
        ----------
        id_values : list or ndarray, shape (n_rows,) or (n_videos,)
            IDs to assign.
        """
        self.data["ids"] = ids

    @property
    def meta_by_frame(self) -> pd.DataFrame:
        """
        Return per-frame metadata columns stored in `self.data`.

        Returns
        -------
        meta_by_frame : pandas.DataFrame, shape (n_rows, n_meta_cols)
        """
        # keep order of columns consistent with self.meta if available
        return self.data[list(self.meta.columns.values)] if not self.meta.empty else pd.DataFrame()

    @meta_by_frame.setter
    def meta_by_frame(self, meta_by_frame: pd.DataFrame) -> None:
        """
        Set per-frame metadata columns into `self.data`.

        Parameters
        ----------
        meta_by_frame : pandas.DataFrame, shape (n_rows, n_meta_cols)
        """
        if meta_by_frame is None or meta_by_frame.empty:
            # nothing to set
            return
        # assign columns by name preserving column order
        self.data.loc[:, list(meta_by_frame.columns.values)] = meta_by_frame.values

    def meta_unique(self, column_id: str) -> List[Any]:
        """
        Return unique values in a metadata column.

        Parameters
        ----------
        column_id : str
            Column name in `self.data`.

        Returns
        -------
        uniques : list
            Unique values in the column (order not guaranteed).
        """
        return list(set(list(self.data[column_id].values)))

    @property
    def n_frames(self) -> int:
        """
        Number of rows/frames in `self.data`.

        Returns
        -------
        n_frames : int
        """
        return int(self.data.shape[0])

    @property
    def embed_vals(self) -> npt.NDArray[Any]:
        """
        Return embedded values stored in column 'embed_vals'.

        Returns
        -------
        embed_vals : ndarray, shape (n_rows, n_components)
            If column absent or empty, this may raise or return an empty array.
        """
        # convert object-list column to ndarray
        return np.array(list(self.data["embed_vals"].to_numpy()))

    @embed_vals.setter
    def embed_vals(self, embed_vals: Optional[npt.NDArray[Any]] = None) -> None:
        """
        Set the 'embed_vals' column from an array-like.

        Parameters
        ----------
        embed_vals : ndarray, shape (n_rows, n_components)
            Embedding coordinates for each row.
        """
        self.data["embed_vals"] = list(embed_vals)

    @property
    def features(self) -> npt.NDArray[Any]:
        """
        Return feature vectors stored in column 'features'.

        Returns
        -------
        features : ndarray, shape (n_rows, n_features)
        """
        return np.array(list(self.data["features"].to_numpy()))

    @features.setter
    def features(self, features: npt.NDArray[Any]) -> None:
        """
        Set the 'features' column from an array-like.

        Parameters
        ----------
        features : ndarray, shape (n_rows, n_features)
        """
        self.data["features"] = list(features)

    @property
    def feat_shape(self) -> Tuple[int, ...]:
        """
        Shape of the features array (n_rows, n_features).

        Returns
        -------
        shape : tuple of ints
        """
        return np.shape(self.features)

    def write_pickle(self, out_path: str) -> None:
        """
        Write this DataStruct to a pickle file named `<out_path>/datastruct.p`.

        Parameters
        ----------
        out_path : str
            Directory path where the pickle will be written. Directory will be created if needed.
        """
        import pickle

        Path(out_path).mkdir(parents=True, exist_ok=True)
        pickle.dump(self, open("".join([out_path, "datastruct.p"]), "wb"))
        return

#TODO: Separately downsamples by id
    def downsample(self, downsample: int) -> "DataStruct":
        """
        Return a downsampled view of this DataStruct.

        Parameters
        ----------
        downsample : int
            Step size for downsampling (e.g., 2 returns every-other frame).

        Returns
        -------
        DataStruct
            New DataStruct with every `downsample`-th row selected.
        """
        # delegate to __getitem__ which uses pandas .loc semantics
        return self[::downsample]
