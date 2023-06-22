import scipy as scp

# from scipy.ndimage import median_filter, convolve
from scipy.spatial.transform import Rotation as R
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import IncrementalPCA
from typing import Optional, Union, List, Tuple, Type
import pandas as pd
from tqdm import tqdm
from scipy.interpolate import CubicSpline, splprep, splev
# import visualization as vis
import functools

def pose_by_id(func):
    @functools.wraps(func)
    def wrapper(pose, id, **kwargs):
        for _, i in enumerate(tqdm(np.unique(id))):
            pose_exp = pose[id == i,:,:]
            pose[id == i ,:,:] = func(pose_exp, **kwargs)
        return pose
    return wrapper

# @pose_by_id
# def align_floor_by_id(pose: np.ndarray,
#     # id:Union[np.ndarray, List],
#     foot_id: Optional[int] = 12,
#     head_id: Optional[int] = 0,
#     dtype: Optional[Type[Union[np.float32, np.float64]]] = np.float32,):
#     return align_floor(pose = pose,foot_id = foot_id, head_id=head_id, dtype=dtype)



def vel_filter(
    pose,
    exp_id,
    threshold: float = 20,
    #    max_iter: int=10,
    connectivity=None,
):
    print("Completing cubic spline interpolation based on velocity")
    for _, i in enumerate(tqdm(np.unique(exp_id))):
        pose_exp = pose[exp_id == i, ...]

        counter = 0
        # while True:
        dxyz = get_frame_diff(pose_exp, time=1, idx_center=False)
        avg_vel = np.linalg.norm(np.sum(dxyz, axis=-1), axis=-1)
        # plt.hist(avg_vel,bins=1000)
        # plt.savefig('./vel_filter.png')
        # import pdb; pdb.set_trace()

        # vis.skeleton_vid3D_features(
        #     pose_exp,
        #     avg_vel,
        #     connectivity,
        #     frames=[np.argmax(avg_vel)],
        #     N_FRAMES=100,
        #     fps=90,
        #     dpi=100,
        #     VID_NAME="pre_vel_filter.mp4",
        #     SAVE_ROOT="./",
        # )

        if np.any(avg_vel > threshold):
            print("vel found true")
            bad_tracking_frames = np.where(avg_vel > threshold)[0]
            print(bad_tracking_frames)
            good_tracking_frames = np.where(avg_vel <= threshold)[0]
            cs = CubicSpline(good_tracking_frames, pose_exp[good_tracking_frames, ...])
            pose_exp[bad_tracking_frames, ...] = cs(bad_tracking_frames)
            #     counter+=1
            #     if counter>=max_iter:
            #         break
            # else:
            #     break
        pose[exp_id == i, ...] = pose_exp

    return pose


def z_filter(pose: np.ndarray, exp_id: Union[np.ndarray, List], threshold: float = 2500, connectivity=None):
    """
    Uses the z value to
    """
    print("Completing cubic spline interpolation based on z values")
    for _, i in enumerate(tqdm(np.unique(exp_id))):
        pose_exp = pose[exp_id == i, ...]

        z_trace = np.sum(pose_exp[..., 2], axis=-1)
        # plt.hist(z_trace,bins=1000)
        # plt.savefig('./z_filter.png')
        # plt.close()
        # import pdb; pdb.set_trace()

        if np.any(z_trace > threshold):
            bad_tracking_frames = np.where(z_trace > threshold)[0]
            print(bad_tracking_frames)
            good_tracking_frames = np.where(z_trace <= threshold)[0]
            cs = CubicSpline(good_tracking_frames, pose_exp[good_tracking_frames, ...])
            pose_exp[bad_tracking_frames, ...] = cs(bad_tracking_frames)

        # z_trace_post = np.sum(pose_exp[...,2],axis=-1)
        # plt.hist(z_trace_post,bins=1000)
        # plt.savefig('./z_filter_post.png')
        # plt.close()
        # import pdb; pdb.set_trace()
        pose[exp_id == i, ...] = pose_exp

        # vis.skeleton_vid3D_features(
        #     pose,
        #     z_trace,
        #     connectivity,
        #     frames=[np.argmax(z_trace)],
        #     N_FRAMES=100,
        #     fps=90,
        #     dpi=100,
        #     VID_NAME="prefilter.mp4",
        #     SAVE_ROOT="./",
        # )

        # vis.skeleton_vid3D_features(
        #     pose_exp,
        #     z_trace_post,
        #     connectivity,
        #     frames=[np.argmax(z_trace)],
        #     N_FRAMES=100,
        #     fps=90,
        #     dpi=100,
        #     VID_NAME="postfilter.mp4",
        #     SAVE_ROOT="./",
        # )

    return pose


def median_filter(pose: np.ndarray, id:Union[np.ndarray, List], filter_len: int = 5):
    """_summary_

    Parameters
    ----------
    pose : np.ndarray
        _description_
    id : Union[np.ndarray, List]
        _description_
    filter_len : int, optional
        _description_, by default 5

    Returns
    -------
    _type_
        _description_
    """
    print("Applying Median Filter")
    for _, i in enumerate(tqdm(np.unique(id))):
        pose_exp = pose[id == i, :, :]
        pose[id == i, :, :] = scp.ndimage.median_filter(
            pose_exp, (filter_len, 1, 1)
        )

    return pose


def anipose_med_filt(
    pose: np.ndarray,
    exp_id: Union[List, np.ndarray],
    filter_len: int = 6,
    threshold: float = 5,
):

    for _, i in enumerate(tqdm(np.unique(exp_id))):
        pose_exp = pose[exp_id == i, :, :]
        # dxyz = get_frame_diff(pose_exp, time=1, idx_center=False)
        # vel =
        pose_error = pose_exp - scp.ndimage.median_filter(
            pose_exp, (filter_len, 1, 1)
        )  # Median filter 5 frames repeat the ends of video
        pose_error = np.linalg.norm(pose_error, axis=-1).mean(axis=-1)

        plt.hist(pose_error, bins=1000)
        plt.savefig("../../results/interp_ensemble/err_hist" + str(i) + ".png")
        plt.close()

        bad_tracking_frames = np.where(pose_error > threshold)[0]
        print(bad_tracking_frames.shape)
        good_tracking_frames = np.where(pose_error <= threshold)[0]
        for joint in tqdm(np.arange(pose_exp.shape[1])):
            for ax in np.arange(pose_exp.shape[2]):
                cs = CubicSpline(
                    good_tracking_frames, pose_exp[good_tracking_frames, joint, ax]
                )
                pose_exp[bad_tracking_frames, joint, ax] = cs(bad_tracking_frames)

        pose[exp_id == i, :, :] = pose_exp

        pose_error = pose_exp - scp.ndimage.median_filter(
            pose_exp, (filter_len, 1, 1)
        )  # Median filter 5 frames repeat the ends of video
        pose_error = np.linalg.norm(pose_error, axis=-1).mean(axis=-1)

        plt.hist(pose_error, bins=1000)
        plt.savefig("../../results/interp_ensemble/err_hist_post" + str(i) + ".png")
        plt.close()

    return pose


def get_frame_diff(x: np.ndarray, time: int, idx_center: bool = True):
    """
    IN:
        x: Numpy array where first axis is time
        time: Size of window to calculate
        idx_center: if `True`, calculates diff centered around point (idx+time - idx-time),
                    if `False`, calculates diff as time before
    """
    prev_x = np.append(np.repeat(x[None, 0, ...], time, axis=0), x[:-time, ...], axis=0)
    if idx_center:
        next_x = np.append(
            x[time:, ...], np.repeat(x[None, -1, ...], time, axis=0), axis=0
        )
        diff = next_x - prev_x
    else:
        diff = x - prev_x

    return diff

@pose_by_id
def align_floor(
    pose: np.ndarray,
    foot_id: Optional[int] = 12,
    head_id: Optional[int] = 0,
    dtype: Optional[Type[Union[np.float32, np.float64]]] = np.float32,
):
    """
    Due to calibration, predictions may be rotated on different axes
    Rotates floor to same x-y plane per video
    IN:
        pose: 3d matrix of (#frames x #joints x #coords)
        exp_id: Video ids per frame
        foot_id: ID of foot to find floor
    OUT:
        pose_rot: Floor aligned poses (#frames x #joints x #coords)
    """
    print("Fitting and rotating the floor for each video to alignment ... ")
    # scp.ndimage.median_filter(pose_exp,(filter_len,1,1))

    # Initial calculation of plane to find outlier values
    [xy, z] = [pose[:, foot_id, :2], pose[:, foot_id, 2]]
    const = np.ones((pose.shape[0], 1))
    coeff = np.linalg.lstsq(np.append(xy, const, axis=1), z, rcond=None)[0]
    z_diff = (
        pose[:, foot_id, 0] * coeff[0]
        + pose[:, foot_id, 1] * coeff[1]
        + coeff[2]
    ) - pose[:, foot_id, 2]
    z_mean = np.mean(z_diff)
    z_range = np.std(z_diff) * np.float32(1.5)
    mid_foot_vals = np.where(
        (z_diff > z_mean - z_range) & (z_diff < z_mean + z_range)
    )[
        0
    ]  # Removing outlier values of foot

    # Recalculating plane with outlier values removed
    [xy, z] = [
        pose[mid_foot_vals, foot_id, :2],
        pose[mid_foot_vals, foot_id, 2],
    ]
    const = np.ones((xy.shape[0], 1))
    coeff = np.linalg.lstsq(np.append(xy, const, axis=1), z, rcond=None)[0]

    # Calculating rotation matrices
    un = np.array([-coeff[0], -coeff[1], 1]) / np.linalg.norm(
        [-coeff[0], -coeff[1], 1]
    )
    vn = np.array([0, 0, 1])
    theta = np.arccos(np.clip(np.dot(un, vn), -1, 1))
    rot_vec = np.cross(un, vn) / np.linalg.norm(np.cross(un, vn)) * theta
    rot_mat = R.from_rotvec(rot_vec).as_matrix().astype(dtype)
    rot_mat = np.expand_dims(rot_mat, axis=2).repeat(
        pose.shape[0] * pose.shape[1], axis=2
    )
    pose[:, :, 2] -= coeff[2]  # Fixing intercept to zero
    # Rotating
    pose_rot = np.einsum(
        "jki,ik->ij", rot_mat, np.reshape(pose, (-1, 3))
    ).reshape(pose.shape)

    ## Checking to make sure snout is on average above the feet
    assert np.mean(pose_rot[:, head_id, 2]) > np.mean(
        pose_rot[:, foot_id, 2]
    )  # checking head is above foot

    return pose_rot

def align_floor_by_id(
    pose: np.ndarray,
    exp_id: Union[List, np.ndarray],
    foot_id: Optional[int] = 12,
    head_id: Optional[int] = 0,
    dtype: Optional[Type[Union[np.float32, np.float64]]] = np.float32,
    plot_folder: Optional[str] = None,
):
    """
    Due to calibration, predictions may be rotated on different axes
    Rotates floor to same x-y plane per video
    IN:
        pose: 3d matrix of (#frames x #joints x #coords)
        exp_id: Video ids per frame
        foot_id: ID of foot to find floor
    OUT:
        pose_rot: Floor aligned poses (#frames x #joints x #coords)
    """
    pose_rot = pose
    print("Fitting and rotating the floor for each video to alignment ... ")
    for _, i in enumerate(tqdm(np.unique(exp_id))):  # Separately for each video
        pose_exp = pose[exp_id == i, :, :]
        # scp.ndimage.median_filter(pose_exp,(filter_len,1,1))

        # Initial calculation of plane to find outlier values
        [xy, z] = [pose_exp[:, foot_id, :2], pose_exp[:, foot_id, 2]]
        const = np.ones((pose_exp.shape[0], 1))
        coeff = np.linalg.lstsq(np.append(xy, const, axis=1), z, rcond=None)[0]
        z_diff = (
            pose_exp[:, foot_id, 0] * coeff[0]
            + pose_exp[:, foot_id, 1] * coeff[1]
            + coeff[2]
        ) - pose_exp[:, foot_id, 2]
        z_mean = np.mean(z_diff)
        z_range = np.std(z_diff) * np.float32(1.5)
        mid_foot_vals = np.where(
            (z_diff > z_mean - z_range) & (z_diff < z_mean + z_range)
        )[
            0
        ]  # Removing outlier values of foot

        # Recalculating plane with outlier values removed
        [xy, z] = [
            pose_exp[mid_foot_vals, foot_id, :2],
            pose_exp[mid_foot_vals, foot_id, 2],
        ]
        const = np.ones((xy.shape[0], 1))
        coeff = np.linalg.lstsq(np.append(xy, const, axis=1), z, rcond=None)[0]

        # Calculating rotation matrices
        un = np.array([-coeff[0], -coeff[1], 1]) / np.linalg.norm(
            [-coeff[0], -coeff[1], 1]
        )
        vn = np.array([0, 0, 1])
        theta = np.arccos(np.clip(np.dot(un, vn), -1, 1))
        rot_vec = np.cross(un, vn) / np.linalg.norm(np.cross(un, vn)) * theta
        rot_mat = R.from_rotvec(rot_vec).as_matrix().astype(dtype)
        rot_mat = np.expand_dims(rot_mat, axis=2).repeat(
            pose_exp.shape[0] * pose_exp.shape[1], axis=2
        )
        pose_exp[:, :, 2] -= coeff[2]  # Fixing intercept to zero
        # Rotating
        pose_rot[exp_id == i, :, :] = np.einsum(
            "jki,ik->ij", rot_mat, np.reshape(pose_exp, (-1, 3))
        ).reshape(pose_exp.shape)

        if plot_folder:
            xx, yy = np.meshgrid(range(-300, 300, 10), range(-300, 300, 10))
            zz = coeff[0] * xx + coeff[1] * yy + coeff[2]
            fig = plt.figure(figsize=(20, 20))
            ax = plt.axes(projection="3d")
            # ax.scatter3D(pose_exp[1000,foot_id,0], pose_exp[1000,foot_id,1], pose_exp[1000,foot_id,2],s=1000,c='r')
            ax.scatter3D(
                pose_exp[:, foot_id, 0],
                pose_exp[:, foot_id, 1],
                pose_exp[:, foot_id, 2],
                s=1,
            )
            ax.plot_surface(xx, yy, zz, alpha=0.2)
            plt.savefig("".join([plot_folder, "/before_rot", str(i), ".png"]))
            plt.close()

            fig = plt.figure(figsize=(20, 20))
            ax = plt.axes()
            ax.scatter(
                pose_rot[exp_id == i, foot_id, 0],
                pose_rot[exp_id == i, foot_id, 2],
                s=1,
            )
            # ax.scatter(pose_rot[1000,foot_id,0], pose_rot[1000,foot_id,2],s=100,c='r')
            plt.savefig("./after_rot", str(i), ".png")
            plt.close()

        ## Checking to make sure snout is on average above the feet
        assert np.mean(pose_rot[exp_id == i, head_id, 2]) > np.mean(
            pose_rot[exp_id == i, foot_id, 2]
        )  # checking head is above foot

    return pose_rot


def center_spine(pose, keypt_idx=4):
    print("Centering poses to mid spine ...")
    # Center spine_m to (0,0,0)
    return pose - np.expand_dims(pose[:, keypt_idx, :], axis=1)


def rotate_spine(pose, keypt_idx=[4, 3], lock_to_x=False,
                 dtype: Optional[Type[Union[np.float32, np.float64]]] = np.float32):
    """
    Centers mid spine to (0,0,0) and aligns spine_m -> spine_f to x-z plane
    IN:
        pose: 3d matrix of (#frames x #joints x #coords)
        keypt_idx: List [spine_m idx, spine_f idx]
        lock_to_x: Also rotates so that spine_m -> spine_f is locked to the x-axis
    OUT:
        pose_rot: Centered and rotated pose (#frames x #joints x #coords)
    """
    num_joints = pose.shape[1]
    yaw = -np.arctan2(
        pose[:, keypt_idx[1], 1], pose[:, keypt_idx[1], 0]
    )  # Find angle to rotate to axis

    if lock_to_x:
        print("Rotating spine to x axis ... ")
        pitch = np.arctan2(pose[:, keypt_idx[1], 2], pose[:, keypt_idx[1], 0])
    else:
        print("Rotating spine to xz plane ... ")
        pitch = np.zeros(yaw.shape)

    # Rotation matrix for pitch and yaw
    rot_mat = np.array(
        [
            [np.cos(yaw) * np.cos(pitch), -np.sin(yaw), np.cos(yaw) * np.sin(pitch)],
            [np.sin(yaw) * np.cos(pitch), np.cos(yaw), np.sin(yaw) * np.sin(pitch)],
            [-np.sin(pitch), np.zeros(len(yaw)), np.cos(pitch)],
        ]
    ).repeat(num_joints, axis=2)
    pose_rot = np.einsum("jki,ik->ij", rot_mat, np.reshape(pose, (-1, 3))).reshape(
        pose.shape
    )

    # Making sure Y value of spine f doesn't deviate much from 0
    assert (
        pose_rot[:, keypt_idx[1], 1].max() < 1e-5
        and pose_rot[:, keypt_idx[1], 1].min() > -1e-5
    )
    if lock_to_x:  # Making sure Z value of spine f doesn't deviate much from 0
        assert (
            pose_rot[:, keypt_idx[1], 2].max() < 1e-5
            and pose_rot[:, keypt_idx[1], 2].min() > -1e-5
        )

    return pose_rot.astype(dtype)


def get_lengths(pose, linkages):
    """
    Get lengths of all linkages
    """
    print("Calculating length of all linkages ... ")
    linkages = np.array(linkages)
    lengths = np.square(pose[:, linkages[:, 1], :] - pose[:, linkages[:, 0], :])
    lengths = np.sum(np.sqrt(lengths), axis=2)
    return lengths


def rolling_window(data, window):
    """
    Returns a view of data windowed (data.shape, window)
    Pads the ends with the edge values
    """
    try:
        assert window % 2 == 1
    except ValueError:
        print("Window size must be odd")
        raise

    # Padding frames with the edge values with (window size/2 - 1)
    pad = int(np.floor(window / 2))
    d_pad = np.pad(data, ((pad, pad), (0, 0)), mode="edge").T
    shape = d_pad.shape[:-1] + (d_pad.shape[-1] - pad * 2, window)
    strides = d_pad.strides + (d_pad.strides[-1],)
    return np.swapaxes(
        np.lib.stride_tricks.as_strided(d_pad, shape=shape, strides=strides), 0, 1
    )


def get_velocities(
    pose,
    exp_id,
    joint_names,
    joints=[0, 3, 5],
    widths=[3, 31, 89],
    abs_val=False,
    sample_freq=90,
):
    """
    Returns absolute velocity, as well as x, y, and z velocities over varying widths
    Takes mean velocity over rolling window of width
    Also returns the standard deviation of these velocities over varying widths
    IN:
        pose: Non-centered and and optional rotated pose (#frames, #joints, #xyz)
        exp_id: Video ids per frame
        joints: joints to calculate absolute velocities
        widths: Number of frames to average velocity over (must be odd)
        sample_freq: Sampling frequency of the videos
    OUT:
        vel: velocity features (#frames x #joints*#widths)
    """
    if np.any(np.sum(pose, axis=(0, 2)) == 0):
        print("Detected centered pose input - calculating relative velocities ... ")
        tag = "rel"
    else:
        print("Calculating absolute velocities ... ")
        tag = "abs"

    ax_labels = ["norm", "x", "y", "z"]
    vel = np.zeros((pose.shape[0], len(joints) * len(widths) * len(ax_labels)))
    vel_stds = np.zeros(vel.shape)
    vel_labels, std_labels = [], []

    for _, i in enumerate(tqdm(np.unique(exp_id))):  # Separating by video
        pose_exp = pose[exp_id == i, :, :][:, joints, :]

        # Calculate distance beetween  times t - (t-1)
        prev_pose = np.append(pose_exp[None, 0, :, :], pose_exp[:-1, :, :], axis=0)
        dxyz = pose_exp - prev_pose  # distance for each axis

        # Appending Euclidean vector magnitude of distance and multiplying by sample_freq to get final velocities
        dv = (
            np.append(np.linalg.norm(dxyz, axis=-1)[:, :, None], dxyz, axis=-1)
            * sample_freq
        )
        dv = np.reshape(dv, (pose_exp.shape[0], -1))
        if abs_val:
            dv = np.abs(dv)
        # Calculate average velocity and velocity stds over the windows
        for j, width in enumerate(widths):
            kernel = np.ones((width, 1)) / width
            vel[
                exp_id == i, j * len(joints) * 4 : (j + 1) * len(joints) * 4
            ] = scp.ndimage.convolve(dv, kernel, mode="constant")
            vel_stds[
                exp_id == i, j * len(joints) * 4 : (j + 1) * len(joints) * 4
            ] = np.std(rolling_window(dv, width), axis=-1)

            if i == np.unique(exp_id)[0]:
                vel_labels += [
                    "_".join([tag, "vel", ax, joint_names[joint], str(width)])
                    for joint in joints
                    for ax in ax_labels
                ]
                std_labels += [
                    "_".join([tag, "vel_std", ax, joint_names[joint], str(width)])
                    for joint in joints
                    for ax in ax_labels
                ]

    # vel_feats = pd.DataFrame(np.hstack((vel,vel_stds)), columns=vel_labels+std_labels)
    return np.hstack((vel, vel_stds)), vel_labels + std_labels


def get_velocities_fast(
    pose,
    exp_id,
    joint_names,
    joints=[0, 3, 5],
    widths=[1, 15, 45],
    abs_val=False,
    sample_freq=90,
    std=True,
):
    """
    Returns absolute velocity, as well as x, y, and z velocities over varying widths
    Takes distance at (t+width) from (t-width)
    Also returns the standard deviation of these velocities over varying widths
    IN:
        pose: Non-centered and and optional rotated pose (#frames, #joints, #xyz)
        exp_id: Video ids per frame
        joints: joints to calculate absolute velocities
        widths: Number of frames to average velocity over (must be odd)
        sample_freq: Sampling frequency of the videos
    OUT:
        vel: velocity features (#frames x #joints*#widths)
    """
    if np.any(np.sum(pose, axis=(0, 2)) == 0):
        print("Detected centered pose input - calculating relative velocities ... ")
        tag = "rel"
    else:
        print("Calculating absolute velocities ... ")
        tag = "abs"

    ax_labels = ["norm", "x", "y", "z"]
    vel = np.zeros((pose.shape[0], len(joints) * len(widths) * len(ax_labels)))
    vel_stds = np.zeros(vel.shape)
    vel_labels, std_labels = [], []

    for _, i in enumerate(tqdm(np.unique(exp_id))):  # Separating by video
        pose_exp = pose[exp_id == i, :, :][:, joints, :]

        # Calculate average velocity and velocity stds over the windows
        for j, width in enumerate(widths):
            # Calculate distance beetween  times t - (t-1)
            dxyz = get_frame_diff(pose_exp, time=width, idx_center=True)

            # Appending Euclidean vector magnitude of distance and multiplying by sample_freq to get final velocities
            dv = (
                np.append(np.linalg.norm(dxyz, axis=-1)[:, :, None], dxyz, axis=-1)
                * sample_freq
                / (width * 2 + 1)
            )
            dv = np.reshape(dv, (pose_exp.shape[0], -1))

            vel[exp_id == i, j * len(joints) * 4 : (j + 1) * len(joints) * 4] = dv

            if i == np.unique(exp_id)[0]:
                vel_labels += [
                    "_".join([tag, "vel", ax, joint_names[joint], str(2 * width + 1)])
                    for joint in joints
                    for ax in ax_labels
                ]

        if std:
            dxyz = get_frame_diff(pose_exp, time=1, idx_center=False)
            # import pdb; pdb.set_trace()
            dv = (
                np.append(np.linalg.norm(dxyz, axis=-1)[:, :, None], dxyz, axis=-1)
                * sample_freq
            )
            dv = np.reshape(dv, (pose_exp.shape[0], -1))
            for j, width in enumerate(widths):
                # if i==49:
                #     import pdb; pdb.set_trace()
                vel_stds[
                    exp_id == i, j * len(joints) * 4 : (j + 1) * len(joints) * 4
                ] = np.std(rolling_window(dv, 2 * width + 1), axis=-1)

                if i == np.unique(exp_id)[0]:
                    std_labels += [
                        "_".join(
                            [tag, "vel_std", ax, joint_names[joint], str(2 * width + 1)]
                        )
                        for joint in joints
                        for ax in ax_labels
                    ]

    # vel_feats = pd.DataFrame(np.hstack((vel,vel_stds)), columns=vel_labels+std_labels)
    if std:
        return np.hstack((vel, vel_stds)), vel_labels + std_labels
    else:
        return vel, vel_labels


def get_ego_pose(pose, joint_names):
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
    # pose_df = pd.DataFrame(pose,columns=labels)
    return pose, labels


def get_euler_angles(pose, link_pairs):
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
    angles = np.zeros((pose.shape[0], len(link_pairs), 3))
    feat_labels = []
    plane_dict = {"xy": [0, 1], "xz": [0, 2], "yz": [1, 2]}
    for i, pair in enumerate(tqdm(link_pairs)):
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


def get_angles(pose, link_pairs):
    angles = np.zeros((pose.shape[0], len(link_pairs)))
    feat_labels = []
    print("Calculating joint angles ... ")
    for i, pair in enumerate(tqdm(link_pairs)):
        v1 = pose[:, pair[0], :] - pose[:, pair[1], :]  # Calculate vectors
        v2 = pose[:, pair[2], :] - pose[:, pair[1], :]
        # import pdb; pdb.set_trace()
        v1_u = v1 / np.linalg.norm(v1, axis=1)[..., None] # Unit vectors
        v2_u = v2 / np.linalg.norm(v2, axis=1)[..., None]

        angles[:, i] = np.arccos(np.clip(np.sum(v1_u * v2_u, axis=1), -1, 1))

        feat_labels += ["_".join(["ang"] + [str(i) for i in pair])]
    return angles, feat_labels


def get_angular_vel(angles, angle_labels, exp_id, widths=[3, 31, 89], sample_freq=90):
    """
    Calculates angular velocity of previously defined angles
    IN:
        angles: Pandas dataframe of angles ()
    """
    print("Calculating velocities of angles ... ")
    num_ang = angles.shape[1]
    avel = np.zeros((angles.shape[0], num_ang * len(widths)))
    avel_stds = np.zeros(avel.shape)
    avel_labels, std_labels = [], []
    for _, i in enumerate(tqdm(np.unique(exp_id))):
        ang_exp = angles[exp_id == i, :]
        prev_ang = np.append(ang_exp[None, 0, :], ang_exp[:-1, :], axis=0)
        dtheta = (ang_exp - prev_ang) * sample_freq
        for j, width in enumerate(widths):
            kernel = np.ones((width, 1)) / width
            avel[exp_id == i, j * num_ang : (j + 1) * num_ang] = scp.ndimage.convolve(
                dtheta, kernel, mode="constant"
            )
            avel_stds[exp_id == i, j * num_ang : (j + 1) * num_ang] = np.std(
                rolling_window(dtheta, width), axis=-1
            )

            if i == np.unique(exp_id)[0]:
                avel_labels += [
                    "_".join([label.replace("ang", "avel"), str(width)])
                    for label in angle_labels
                ]
                std_labels += [
                    "_".join([label.replace("ang", "avel_std"), str(width)])
                    for label in angle_labels
                ]

    # avel_feats = pd.DataFrame(np.hstack((avel,avel_stds)), columns=avel_labels+std_labels)

    return np.hstack((avel, avel_stds)), avel_labels + std_labels


def get_angular_vel_fast(
    angles, angle_labels, exp_id, widths=[1, 15, 45], sample_freq=90
):
    """
    Calculates angular velocity of previously defined angles
    IN:
        angles: Pandas dataframe of angles ()
    """
    print("Calculating velocities of angles ... ")
    num_ang = angles.shape[1]
    avel = np.zeros((angles.shape[0], num_ang * len(widths)))
    avel_stds = np.zeros(avel.shape)
    avel_labels, std_labels = [], []
    for _, i in enumerate(tqdm(np.unique(exp_id))):
        ang_exp = angles[exp_id == i, :]
        for j, width in enumerate(widths):
            dtheta = (
                get_frame_diff(ang_exp, time=width, idx_center=True)
                * sample_freq
                / (width * 2 + 1)
            )
            avel[exp_id == i, j * num_ang : (j + 1) * num_ang] = dtheta

            if i == np.unique(exp_id)[0]:
                avel_labels += [
                    "_".join([label.replace("ang", "avel"), str(2 * width + 1)])
                    for label in angle_labels
                ]
                std_labels += [
                    "_".join([label.replace("ang", "avel_std"), str(2 * width + 1)])
                    for label in angle_labels
                ]

        dtheta = get_frame_diff(ang_exp, time=1, idx_center=False) * sample_freq
        for j, width in enumerate(widths):
            avel_stds[exp_id == i, j * num_ang : (j + 1) * num_ang] = np.std(
                rolling_window(dtheta, 2 * width + 1), axis=-1
            )

    # avel_feats = pd.DataFrame(np.hstack((avel,avel_stds)), columns=avel_labels+std_labels)

    return np.hstack((avel, avel_stds)), avel_labels + std_labels


def get_head_angular(pose, exp_id, widths=[5, 10, 50], link=[0, 3, 4]):
    """
    Getting x-y angular velocity of head
    IN:
        pose: Non-centered, optional rotated pose
    """
    v1 = pose[:, link[0], :2] - pose[:, link[1], :2]
    v2 = pose[:, link[2], :2] - pose[:, link[1], :2]

    angle = np.arctan2(v1[:, 0], v1[:, 1]) - np.arctan2(v2[:, 0], v2[:, 1])
    angle = np.where(angle > 0, angle, angle + 2 * np.pi)

    angular_vel = np.zeros((len(angle), len(widths)))
    for _, i in tqdm.tqdm(np.unique(exp_id)):
        angle_exp = angle[exp_id == i]
        d_angv = angle_exp - np.append(angle_exp[0], angle_exp[:-1])
        for i, width in enumerate(widths):
            kernel = np.ones(width) / width
            angular_vel[exp_id == i, i] = scp.ndimage.convolve(
                d_angv, kernel, mode="constant"
            )

    return angular_vel


def wavelet(
    features, labels, exp_id, sample_freq=90, freq=np.linspace(1, 25, 25), w0=5
):
    # scp.signal.morlet2(500, )
    print("Calculating wavelets ... ")
    widths = w0 * sample_freq / (2 * freq * np.pi)
    wlet_feats = np.zeros((features.shape[0], len(freq) * features.shape[1]))

    wlet_labels = [
        "_".join(["wlet", label, str(np.round(f, 2))]) for label in labels for f in freq
    ]

    for i in np.unique(exp_id):
        print("Calculating wavelets for video " + str(i))
        for j in tqdm(range(features.shape[1])):
            wlet_feats[exp_id == i, j * len(freq) : (j + 1) * len(freq)] = np.abs(
                scp.signal.cwt(
                    features[exp_id == i, j], scp.signal.morlet2, widths, w=w0
                ).T
            )
    return wlet_feats, wlet_labels


def pca(
    features,
    labels,
    categories=["vel", "ego_euc", "ang", "avel"],
    n_pcs=10,
    method="ipca",
):
    print("Calculating principal components ... ")

    # Initializing the PCA method
    if method.startswith("torch"):
        import torch

        pca_feats = torch.zeros(features.shape[0], len(categories) * n_pcs)
        features = torch.tensor(features)
    else:
        # Centering the features if not torch (pytorch does it itself)
        features = features - features.mean(axis=0)
        pca_feats = np.zeros((features.shape[0], len(categories) * n_pcs))

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

        elif method.startswith("torch"):
            feat_cat = features[:, cols_idx]
            if method.endswith("_gpu"):
                feat_cat = feat_cat.cuda()

            if "pca" in method:
                (_, _, V) = torch.pca_lowrank(feat_cat)
            elif "svd" in method:
                feat_cat -= feat_cat.mean()
                (_, _, V) = torch.linalg.svd(feat_cat)

            if method.endswith("_gpu"):
                pca_feats[:, i * n_pcs : (i + 1) * n_pcs] = (
                    torch.matmul(feat_cat, V[:, :n_pcs]).detach().cpu()
                )
                feat_cat.detach().cpu()
                V.detach().cpu()
            else:
                pca_feats[:, i * n_pcs : (i + 1) * n_pcs] = torch.matmul(
                    feat_cat, V[:, :n_pcs]
                )

        elif method == "fbpca":
            (_, _, V) = fbpca.pca(features[:, cols_idx], k=n_pcs)
            pca_feats[:, i * n_pcs : (i + 1) * n_pcs] = np.matmul(
                features[:, cols_idx], V.T
            )

    if method.startswith("torch_pca"):
        pca_feats = pca_feats.numpy()

    # assert num_cols == features.shape[1]

    pc_labels = [
        "_".join([cat, "pc" + str(i)]) for cat in categories for i in range(n_pcs)
    ]

    return pca_feats, pc_labels


def remove_edge_ids(id: np.array, size: int):
    ind = np.arange(len(id))
    unsorted_unique = id[np.sort(np.unique(id, return_index=True)[1])]

    for i, label in enumerate(unsorted_unique):
        if i == 0:
            ind_out = ind[id == label][size:-size]
        else:
            ind_out = np.append(ind_out, ind[id == label][size:-size])

    assert len(ind_out) == len(id) - len(unsorted_unique) * 2 * size

    return ind_out


def standard_scale(features, labels, clip=None):
    features -= features.mean(axis=0)
    feat_std = np.std(features, axis=0)
    features = features[:, feat_std != 0]
    if clip is None:
        features = features / feat_std[feat_std != 0]
    else:
        features = np.clip(features / feat_std[feat_std != 0], -clip, clip)
    labels = [label for i, label in enumerate(labels) if feat_std[i] != 0]

    return features, labels
