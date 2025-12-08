from scipy.ndimage import convolve
import numpy as np
from typing import Optional, Union, List, Tuple, Type
from tqdm import tqdm
from neuroposelib.utils import by_id, rolling_window, get_frame_diff
import pywt
import numpy.typing as npt
from sklearn.preprocessing import StandardScaler

def get_lengths(pose: npt.NDArray, links: npt.ArrayLike) -> npt.NDArray:
    """Get lengths of all defined segments.

    Parametersxwxwxw
    ----------
    pose : npt.NDArray
        Array of 3D pose values of shape (# frames, # keypoints, 3 coordinates).
    links : npt.ArrayLike
        Indices of segment links in pose array (# segments, 2)

    Returns
    -------
    lengths : npt.NDArray
        Array of segment lengths (# frames, # segments)
    """
    print("Calculating length of all linkages ... ")
    lengths = np.square(pose[:, links[:, 1], :] - pose[:, links[:, 0], :])
    lengths = np.sum(np.sqrt(lengths), axis=2)
    return lengths


def get_velocities(
    pose: npt.NDArray,
    ids: npt.ArrayLike,
    joint_names: List[str],
    joints: npt.ArrayLike = [0, 3, 5],
    widths: npt.ArrayLike = [3, 31, 89],
    abs_val: bool = False,
    fs: int = 90,
    std: bool = True,
) -> tuple[npt.NDArray, List[str]]:
    """Returns absolute velocity, as well as x, y, and z velocities over varying widths
    Takes distance at (t+width) from (t-width).
    Also returns the standard deviation of these velocities over varying widths.

    Parameters
    ----------
    pose : npt.NDArray
        Array of 3D pose values of shape (# frames, # keypoints, 3 coordinates).
    ids : npt.ArrayLike
        Id label for each frame in pose, e.g. video id (# frames).
    joint_names : List[str]
        List of labels for keypoints/joints ordered by the indices in the pose array.
    joints : npt.ArrayLike, optional
        Keypoints/joints for which to calculate speed.
    widths : npt.ArrayLike, optional
        Size of windows for average speed calculation (must be odd).
    abs_val : bool, optional
        If True, will output the absolute value of the per frame displacement.
    fs : int, optional
        Frame rate per second.
    std : bool, optional
        If True, will calculate speed standard deviations within windows.

    Returns
    -------
    velocity: npt.NDArray
        Array of velocity features per frame (# frames, # velocity features)
    labels: List[str]]
        List of labels for velocity features in columns of velocity array.
    """
    if np.any(np.sum(pose, axis=(0, 2)) == 0):
        print("Detected centered pose input - calculating relative velocities ... ")
        tag = "rel"
    else:
        print("Calculating absolute velocities ... ")
        tag = "abs"

    dtype = pose.dtype
    ax_labels = ["norm", "x", "y", "z"]
    vel = np.zeros(
        (pose.shape[0], len(joints) * len(widths) * len(ax_labels)), dtype=dtype
    )
    vel_stds = np.zeros(vel.shape, dtype=dtype)
    vel_labels, std_labels = [], []

    for _, i in enumerate(tqdm(np.unique(ids))):  # Separating by video
        pose_exp = pose[ids == i, ...][:, joints, :]

        # Calculate average velocity and velocity stds over the windows
        for j, width in enumerate(widths):
            # Calculate distance beetween  times t - (t-1)
            dxyz = get_frame_diff(pose_exp, time=width, idx_center=True)

            # Appending Euclidean vector magnitude of distance and multiplying by sample_freq to get final velocities
            dv = (
                np.append(np.linalg.norm(dxyz, axis=-1)[:, :, None], dxyz, axis=-1)
                * fs
                / (width * 2 + 1)
            )
            dv = np.reshape(dv, (pose_exp.shape[0], -1))

            if abs_val:
                dv = np.abs(dv)

            vel[ids == i, j * len(joints) * 4 : (j + 1) * len(joints) * 4] = dv

            if i == np.unique(ids)[0]:
                vel_labels += [
                    "_".join([tag, "vel", ax, joint_names[joint], str(2 * width + 1)])
                    for joint in joints
                    for ax in ax_labels
                ]

        if std:
            dxyz = get_frame_diff(pose_exp, time=1, idx_center=False)
            # import pdb; pdb.set_trace()
            dv = np.append(np.linalg.norm(dxyz, axis=-1)[..., None], dxyz, axis=-1) * fs
            dv = np.reshape(dv, (pose_exp.shape[0], -1))
            if abs_val:
                dv = np.abs(dv)
            for j, width in enumerate(widths):
                # if i==49:
                #     import pdb; pdb.set_trace()
                vel_stds[ids == i, j * len(joints) * 4 : (j + 1) * len(joints) * 4] = (
                    np.std(rolling_window(dv, 2 * width + 1), axis=-1)
                )

                if i == np.unique(ids)[0]:
                    std_labels += [
                        "_".join(
                            [tag, "vel_std", ax, joint_names[joint], str(2 * width + 1)]
                        )
                        for joint in joints
                        for ax in ax_labels
                    ]

    if std:
        return np.hstack((vel, vel_stds)), vel_labels + std_labels
    else:
        return vel, vel_labels


def get_ego_pose(
    pose: npt.NDArray, joint_names: List[str]
) -> tuple[npt.NDArray, List[str]]:
    """Takes centered, forward-rotated pose array and reshapes to tabular data.

    Parameters
    ----------
    pose : npt.NDArray
        Array of 3D pose values of shape (# frames, # keypoints, 3 coordinates).
    joint_names : List[str]
        List of labels for keypoints/joints ordered by the indices in the pose array.

    Returns
    -------
    ego_pose: npt.NDArray
        Array of egocentric pose coordinates per frame (# frames, # keypoints * 3)
    labels: List[str]]
        List of labels for pose features in columns of ego_pose array.

    Raises
    ------
    ValueError
        If input pose is not centered or rotated to be facing forward.
    """
    print("Reformatting pose to egocentric pose features ... ")
    is_centered = np.any(np.sum(pose, axis=(0, 2)) == 0)
    is_rotated = np.any(
        np.logical_and(
            np.sum(pose[:, :, 1], axis=0) < 1e-8, np.sum(pose[:, :, 1], axis=0) > -1e-8
        )
    )
    if not (is_centered and is_rotated):
        raise ValueError("Pose must be centered and rotated")

    pose = np.reshape(pose, (pose.shape[0], pose.shape[1] * pose.shape[2]))
    axis = ["x", "y", "z"]
    labels = ["_".join(["ego_euc", joint, ax]) for joint in joint_names for ax in axis]

    return pose, labels


def get_euler_angles(
    pose: npt.NDArray, links: npt.ArrayLike
) -> tuple[npt.NDArray, List[str]]:
    """Calculates 3 angles for pairs of segment vectors. Angles calculated are those between projections of each vector onto the 3 xyz planes.

    * Currently using unsigned angles

    Parameters
    ----------
    pose : npt.NDArray
        Array of 3D pose values of shape (# frames, # keypoints, 3 coordinates).
    links : npt.ArrayLike
        List of tuples with 3 points representing two vectors between which to calculate (# segment pairs, 3).

    Returns
    -------
    angles: npt.NDArray
        Array of angle features per frame (# frames, # angles)
    labels: List[str]]
        List of labels for angle features in columns of angles array.
    """
    print("Calculating joint angles - Euler ... ")
    angles = np.zeros((pose.shape[0], len(links), 3))
    feat_labels = []
    plane_dict = {"xy": [0, 1], "xz": [0, 2], "yz": [1, 2]}
    for i, pair in enumerate(tqdm(links)):
        v1 = pose[:, pair[0], :] - pose[:, pair[1], :]  # Calculate vectors
        v2 = pose[:, pair[2], :] - pose[:, pair[1], :]
        for j, key in enumerate(plane_dict):
            # This is for signed angle
            # angles[:,i,j] = np.arctan2(v1[:,plane_dict[key][0]],v1[:,plane_dict[key][1]]) - \
            #                 np.arctan2(v2[:,plane_dict[key][0]],v2[:,plane_dict[key][1]])

            # This is for unsigned angle
            v1_u = v1[:, plane_dict[key]] / np.expand_dims(
                np.linalg.norm(v1[:, plane_dict[key]], axis=1), axis=1
            )
            v2_u = v2[:, plane_dict[key]] / np.expand_dims(
                np.linalg.norm(v2[:, plane_dict[key]], axis=1), axis=1
            )
            angles[:, i, j] = np.arccos(np.clip(np.sum(v1_u * v2_u, axis=1), -1, 1))

            feat_labels += ["_".join(["ang"] + [str(i) for i in pair] + [key])]

    # Fix all negative angles so that final is between 0 and 2pi
    # round_offset = 1e-4
    # angles = np.clip(angles, -2*np.pi+round_offset, 2*np.pi-round_offset)
    # angles = np.where(angles>0, angles, angles+2*np.pi)

    angles = np.reshape(angles, (angles.shape[0], angles.shape[1] * angles.shape[2]))
    # angles = pd.DataFrame(angles, columns=feat_labels)
    return angles, feat_labels


def get_angles(
    pose: npt.NDArray, links: npt.ArrayLike
) -> tuple[npt.NDArray, List[str]]:
    """Calculates angles between pairs of segment vectors. Angle calculated is the unsigned shortest angle between two vectors.

    Parameters
    ----------
    pose : npt.NDArray
        Array of 3D pose values of shape (# frames, # keypoints, 3 coordinates).
    links : npt.ArrayLike
        List of tuples with 3 points representing two vectors between which to calculate (# angles, 3).

    Returns
    -------
    angles: npt.NDArray
        Array of angle features per frame (# frames, # angles)
    labels: List[str]]
        List of labels for angle features in columns of angles array.
    """
    angles, labels = [], []
    print("Calculating joint angles ... ")
    for i, pair in enumerate(tqdm(links)):
        v1 = pose[:, pair[0], :] - pose[:, pair[1], :]  # Calculate vectors
        v2 = pose[:, pair[2], :] - pose[:, pair[1], :]

        v1_u = v1 / np.linalg.norm(v1, axis=1)[..., None]  # Unit vectors
        v2_u = v2 / np.linalg.norm(v2, axis=1)[..., None]

        angles += [np.arccos(np.clip(np.sum(v1_u * v2_u, axis=1), -1, 1))[..., None]]

        labels += ["_".join(["ang"] + [str(i) for i in pair])]
    angles = np.concatenate(angles, axis=1)
    return angles, labels


def get_angular_vel(
    angles: npt.NDArray,
    angle_labels: List[str],
    ids: npt.ArrayLike,
    widths: npt.ArrayLike = [1, 15, 45],
    fs: int = 90,
) -> tuple[npt.NDArray, List[str]]:
    """Calculates angular velocity of an angular velocity array and standard deviation of velocities over defined windows.

    Parameters
    ----------
    angles : npt.NDArray
        Array of angle features per frame (# frames, # angles)
    angle_labels: List[str]]
        List of labels for angle features in columns of angles array.
    ids : npt.ArrayLike
        Id label for each frame in pose, e.g. video id (# frames).
    widths : widths : npt.ArrayLike, optional
        Size of windows for average speed calculation (must be odd).
    fs : int, optional
        Frame rate per second.

    Returns
    -------
    angular_velocity: npt.NDArray
        Array of angular velocity features per frame (# frames, # angular velocity features)
    labels: List[str]]
        List of labels for velocity features in columns of angular velocity array.
    """
    print("Calculating velocities of angles ... ")
    num_ang = angles.shape[1]
    avel = np.zeros((angles.shape[0], num_ang * len(widths)), dtype=angles.dtype)
    avel_stds = np.zeros(avel.shape, dtype=angles.dtype)
    avel_labels, std_labels = [], []
    for _, i in enumerate(tqdm(np.unique(ids))):
        ang_exp = angles[ids == i, :]
        for j, width in enumerate(widths):
            dtheta = (
                get_frame_diff(ang_exp, time=width, idx_center=True)
                * fs
                / (width * 2 + 1)
            )
            avel[ids == i, j * num_ang : (j + 1) * num_ang] = dtheta

            if i == np.unique(ids)[0]:
                avel_labels += [
                    "_".join([label.replace("ang", "avel"), str(2 * width + 1)])
                    for label in angle_labels
                ]
                std_labels += [
                    "_".join([label.replace("ang", "avel_std"), str(2 * width + 1)])
                    for label in angle_labels
                ]

        dtheta = get_frame_diff(ang_exp, time=1, idx_center=False) * fs
        for j, width in enumerate(widths):
            avel_stds[ids == i, j * num_ang : (j + 1) * num_ang] = np.std(
                rolling_window(dtheta, 2 * width + 1), axis=-1
            )

    # avel_feats = pd.DataFrame(np.hstack((avel,avel_stds)), columns=avel_labels+std_labels)

    return np.hstack((avel, avel_stds)), avel_labels + std_labels


def _get_head_angular(
    pose: npt.NDArray,
    ids: npt.ArrayLike,
    widths: npt.ArrayLike = [5, 10, 50],
    link: npt.ArrayLike = [0, 3, 4],
) -> npt.NDArray:
    """_summary_

    Parameters
    ----------
    pose : npt.NDArray
        _description_
    ids : npt.ArrayLike
        _description_
    widths : npt.ArrayLike, optional
        _description_, by default [5, 10, 50]
    link : npt.ArrayLike, optional
        _description_, by default [0, 3, 4]

    Returns
    -------
    angular velocity: npt.NDArray
        Array of angular velocities of the head.
    """
    """
    Getting x-y angular velocity of head
    IN:
        pose: Non-centered, optional rotated pose
    """
    v1 = pose[:, link[0], :2] - pose[:, link[1], :2]
    v2 = pose[:, link[2], :2] - pose[:, link[1], :2]

    angle = np.arctan2(v1[:, 0], v1[:, 1]) - np.arctan2(v2[:, 0], v2[:, 1])
    angle = np.where(angle > 0, angle, angle + 2 * np.pi)

    angular_vel = np.zeros((len(angle), len(widths)), dtype=pose.dtype)
    for _, i in tqdm.tqdm(np.unique(ids)):
        angle_exp = angle[ids == i]
        d_angv = angle_exp - np.append(angle_exp[0], angle_exp[:-1])
        for i, width in enumerate(widths):
            kernel = np.ones(width) / width
            angular_vel[ids == i, i] = convolve(d_angv, kernel, mode="constant")

    return angular_vel


def wavelet(
    features: npt.NDArray,
    labels: List[str],
    ids: npt.ArrayLike,
    fs: int = 90,
    freq: npt.ArrayLike = np.geomspace(1, 25, 25),
    bw: float = 1.0,
) -> tuple[npt.NDArray, List[str]]:
    """Applies complex Morlet wavlet transform over feature array.

    Built on [PyWavelet](https://pywavelets.readthedocs.io/en/latest/index.html).

    Parameters
    ----------
    features : npt.NDArray
        2D array of features (# frames, # features).
    labels : List[str]
        List of labels for features in columns of features array.
    ids : npt.ArrayLike
        Id label for each frame in pose, e.g. video id (# frames).
    fs : int, optional
        Frame rate per second.
    freq : npt.ArrayLike, optional
        Frequencies to evaluate.
    bw : float, optional
        Bandwidth of complex Morlet wavelet, by default 1.0

    Returns
    -------
    wavelet: npt.NDArray
        Array of wavelet features per frame (# frames, # features * len(freq))
    labels: List[str]]
        List of labels for wavelet features in columns of wavelet array.
    """

    print("Calculating wavelets ... ")
    # widths = (w0 * fs / (2 * freq * np.pi)).astype(features.dtype)
    wlet_feats = np.zeros(
        (features.shape[0], len(freq) * features.shape[1]), features.dtype
    )

    wlet_labels = [
        "_".join(["wlet", label, str(np.round(f, 2))]) for label in labels for f in freq
    ]
    cmor_str = "cmor{:.1f}-{:.1f}".format(bw, 1.0)
    scales = pywt.frequency2scale(cmor_str, freq) * fs
    for i in np.unique(ids):
        print("Calculating wavelets for video " + str(i))
        wlets_i_f = np.abs(
            pywt.cwt(
                features[ids == i],
                scales=scales,
                wavelet=cmor_str,
                # sampling_period=1 / fs,
                method="fft",
                axis=0,
            )[0]
        )

        wlet_feats[ids == i] = np.swapaxes(wlets_i_f, 0, 1).reshape(
            wlets_i_f.shape[1], -1
        )

    return wlet_feats, wlet_labels


def pca(
    features: npt.NDArray,
    labels: List[str],
    categories: List[str] = ["vel", "ego_euc", "ang", "avel"],
    n_pcs: int = 10,
    max_frames: int = 2e7,
    method: str = "fbpca",
) -> tuple[npt.NDArray, List[str]]:
    """Transforms feature array into principal component scores.

    Parameters
    ----------
    features : npt.NDArray
        2D array of features (# frames, # features).
    labels : List[str]
        List of labels for features in columns of features array.
    categories : List[str], optional
        Category labels of features on which to separately calculate PCs.
    n_pcs : int, optional
        Number of principal components.
    max_frames : int, optional
        Maximum number of frames on which to calculate PCs. Will evenly downsample feature array when exceeded.
    method : str, optional
        Method to use in calculating PCs.

    Returns
    -------
    scores: npt.NDArray
        Array of PC transformed features per frame (# frames, # categories * n_pcs)
    labels: List[str]]
        List of labels for PC transformed features in columns of scores array.
    """
    print("Calculating principal components ... ")
    
    features = StandardScaler().fit_transform(features)# - features.mean(axis=0)
    # features /= features.std(axis=0) + 1e-8
    pca_feats = np.zeros(
        (features.shape[0], len(categories) * n_pcs), dtype=features.dtype
    )

    if method == "ipca":
        from sklearn.decomposition import IncrementalPCA

        pca = IncrementalPCA(n_components=n_pcs, batch_size=None)
    elif method.startswith("fbpca"):
        import fbpca

    num_cols = 0
    for i, cat in enumerate(tqdm(categories)):  # Iterate through each feature category
        cat += "_"
        cols_idx = [
            i
            for i, col in enumerate(labels)
            if (col.startswith(cat) or ("_" + cat in col))
        ]
        num_cols += len(cols_idx)

        if method == "ipca" or method == "sklearn_pca":
            # import pdb; pdb.set_trace()
            pca_feats[:, i * n_pcs : (i + 1) * n_pcs] = pca.fit_transform(
                features[:, cols_idx]
            )

        elif method == "fbpca":
            downsample = int(np.ceil(len(features) / max_frames))
            assert downsample > 0
            (_, _, V) = fbpca.pca(
                features[::downsample, cols_idx].astype(np.float64), k=n_pcs
            )
            pca_feats[:, i * n_pcs : (i + 1) * n_pcs] = np.matmul(
                features[:, cols_idx], V.astype(features.dtype).T
            )

    pc_labels = [
        "_".join([cat, "pc" + str(i)]) for cat in categories for i in range(n_pcs)
    ]

    return pca_feats, pc_labels
