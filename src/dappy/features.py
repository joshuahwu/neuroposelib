from scipy.ndimage import convolve
from scipy.signal import morlet2, cwt
import numpy as np
from typing import Optional, Union, List, Tuple, Type
from tqdm import tqdm
from dappy.utils import by_id, rolling_window, get_frame_diff


def get_lengths(pose: np.ndarray, links: np.ndarray):
    """
    Get lengths of all linkages
    """
    print("Calculating length of all linkages ... ")
    lengths = np.square(pose[:, links[:, 1], :] - pose[:, links[:, 0], :])
    lengths = np.sum(np.sqrt(lengths), axis=2)
    return lengths


def get_velocities(
    pose: np.ndarray,
    ids: Union[np.ndarray, List],
    joint_names: List[str],
    joints: List[int] = [0, 3, 5],
    widths=[3, 31, 89],
    abs_val: bool = False,
    f_s: int = 90,
    std: bool = True,
):
    """
    Returns absolute velocity, as well as x, y, and z velocities over varying widths
    Takes distance at (t+width) from (t-width)
    Also returns the standard deviation of these velocities over varying widths
    IN:
        pose: Non-centered and and optional rotated pose (#frames, #joints, #xyz)
        ids: Video ids per frame
        joints: joints to calculate absolute velocities
        widths: Number of frames to average velocity over (must be odd)
        f_s: Sampling frequency of the videos
    OUT:
        vel: velocity features (#frames x #joints*#widths)
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
                * f_s
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
            dv = (
                np.append(np.linalg.norm(dxyz, axis=-1)[..., None], dxyz, axis=-1) * f_s
            )
            dv = np.reshape(dv, (pose_exp.shape[0], -1))
            if abs_val:
                dv = np.abs(dv)
            for j, width in enumerate(widths):
                # if i==49:
                #     import pdb; pdb.set_trace()
                vel_stds[
                    ids == i, j * len(joints) * 4 : (j + 1) * len(joints) * 4
                ] = np.std(rolling_window(dv, 2 * width + 1), axis=-1)

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


def get_ego_pose(pose: np.ndarray, joint_names: List[str]):
    """
    Takes centered spine rotated pose - reshapes and converts to pandas dataframe
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


def get_euler_angles(pose: np.ndarray, links: np.ndarray):
    """
    Calculates 3 angles for pairs of linkage vectors
    Angles calculated are those between projections of each vector onto the 3 xyz planes
    IN:
        pose: Centered and rotated pose (#frames, #joints, #xyz)
        link_pairs: List of tuples with 3 points between which to calculate angles
    OUT:
        angles: returns 3 angles between link pairs

    ** Currently doing unsigned
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


def get_angles(pose: np.ndarray, links: np.ndarray):
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
    angles: np.ndarray,
    angle_labels: List[str],
    ids: Union[np.ndarray, List],
    widths: List[int] = [1, 15, 45],
    f_s: int = 90,
):
    """
    Calculates angular velocity of previously defined angles
    IN:
        angles: Pandas dataframe of angles ()
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
                * f_s
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

        dtheta = get_frame_diff(ang_exp, time=1, idx_center=False) * f_s
        for j, width in enumerate(widths):
            avel_stds[ids == i, j * num_ang : (j + 1) * num_ang] = np.std(
                rolling_window(dtheta, 2 * width + 1), axis=-1
            )

    # avel_feats = pd.DataFrame(np.hstack((avel,avel_stds)), columns=avel_labels+std_labels)

    return np.hstack((avel, avel_stds)), avel_labels + std_labels


def get_head_angular(
    pose: np.ndarray,
    ids: Union[np.ndarray, List],
    widths: Union[List[int], np.ndarray] = [5, 10, 50],
    link: Union[List[int], np.ndarray] = [0, 3, 4],
):
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
    features: np.ndarray,
    labels: List[str],
    ids: Union[np.ndarray, List],
    f_s: int = 90,
    freq: Union[List, np.ndarray] = np.linspace(1, 25, 25),
    w0: float = 5,
):
    # scp.signal.morlet2(500, )
    print("Calculating wavelets ... ")
    widths = (w0 * f_s / (2 * freq * np.pi)).astype(features.dtype)
    wlet_feats = np.zeros(
        (features.shape[0], len(freq) * features.shape[1]), features.dtype
    )

    wlet_labels = [
        "_".join(["wlet", label, str(np.round(f, 2))]) for label in labels for f in freq
    ]

    for i in np.unique(ids):
        print("Calculating wavelets for video " + str(i))
        for j in tqdm(range(features.shape[1])):
            wlet_feats[ids == i, j * len(freq) : (j + 1) * len(freq)] = np.abs(
                cwt(features[ids == i, j], morlet2, widths, w=w0).T
            )
    return wlet_feats, wlet_labels


def pca(
    features: np.ndarray,
    labels: List,
    categories: List[str]=["vel", "ego_euc", "ang", "avel"],
    n_pcs: int = 10,
    downsample: int = 1, 
    method="fbpca",
):
    print("Calculating principal components ... ")

    # Initializing the PCA method
    # if method.startswith("torch"):
    #     import torch

    #     pca_feats = torch.zeros(features.shape[0], len(categories) * n_pcs)
    #     features = torch.tensor(features)
    # else:
    # Centering the features if not torch (pytorch does it itself)
    features = features - features.mean(axis=0)
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

        # elif method.startswith("torch"):
        #     feat_cat = features[:, cols_idx]
        #     if method.endswith("_gpu"):
        #         feat_cat = feat_cat.cuda()

        #     if "pca" in method:
        #         (_, _, V) = torch.pca_lowrank(feat_cat)
        #     elif "svd" in method:
        #         feat_cat -= feat_cat.mean()
        #         (_, _, V) = torch.linalg.svd(feat_cat)

        #     if method.endswith("_gpu"):
        #         pca_feats[:, i * n_pcs : (i + 1) * n_pcs] = (
        #             torch.matmul(feat_cat, V[:, :n_pcs]).detach().cpu()
        #         )
        #         feat_cat.detach().cpu()
        #         V.detach().cpu()
        #     else:
        #         pca_feats[:, i * n_pcs : (i + 1) * n_pcs] = torch.matmul(
        #             feat_cat, V[:, :n_pcs]
        #         )

        elif method == "fbpca":
            import pdb; pdb.set_trace()
            (_, _, V) = fbpca.pca(features[:, cols_idx].astype(np.float64), k=n_pcs)
            pca_feats[:, i * n_pcs : (i + 1) * n_pcs] = np.matmul(
                features[:, cols_idx], V.astype(features.dtype).T
            )

    # if method.startswith("torch_pca"):
    #     pca_feats = pca_feats.numpy()

    # assert num_cols == features.shape[1]

    pc_labels = [
        "_".join([cat, "pc" + str(i)]) for cat in categories for i in range(n_pcs)
    ]

    return pca_feats, pc_labels
