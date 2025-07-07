import scipy.ndimage as scp_ndi
from scipy.interpolate import CubicSpline
import numpy as np
import numpy.typing as npt
import numpy.typing as npt
from neuroposelib.utils import by_id, get_frame_diff
from typing import Optional, Union, List, Type, Tuple
from typing import Optional, Union, List, Type, Tuple
from tqdm import tqdm
from scipy.spatial.transform import Rotation as R


@by_id
def align_floor_by_id(
    pose: npt.NDArray,
    pose: npt.NDArray,
    foot_id: Optional[int] = 12,
    head_id: Optional[int] = 0,
    dtype: Optional[Type[Union[np.float32, np.float64]]] = np.float32,
    **kwargs
):
    """
    Due to the camera calibration, predictions may be rotated to different world coordinates.
    Rotates the floor to same x-y plane per video ID given.

    Parameters
    ----------
    pose : npt.NDArray
        Array of 3D pose values of shape (# frames, # keypoints, 3 coordinates).
    ids : npt.ArrayLike
        Id label for each frame in pose, e.g. video id (# frames).
    foot_id : Optional[int], optional
        Index of the foot keypoint used to fit to the floor plane, by default 12
    head_id : Optional[int], optional
        Index of the head keypoint used double check that foot keypoints are below head keypoints after rotation, by default None
    dtype : Optional[Type[Union[np.DTypeLike]]], optional
        Desired data type of output array.

    Returns
    -------
    pose : npt.NDArray
        Array of floor-aligned 3D pose values of shape (# frames, # keypoints, 3 coordinates).
    """
    return align_floor(pose=pose, foot_id=foot_id, head_id=head_id, dtype=dtype)
    return align_floor(pose=pose, foot_id=foot_id, head_id=head_id, dtype=dtype,**kwargs)


def align_floor(
    pose: npt.NDArray,
    foot_id: Optional[int] = 12,
    head_id: Optional[int] = 0,
    dtype: Optional[Type[Union[np.float32, np.float64]]] = np.float32,
    is_whole_body: Optional[bool] = True,
):
    """
    Due to the camera calibration, predictions may be rotated to different world coordinates.
    Rotates the floor to same x-y plane for one video.

    Parameters
    ----------
    pose : npt.NDArray
        Array of 3D pose values of shape (# frames, # keypoints, 3 coordinates).
    foot_id : Optional[int], optional
        Index of the foot keypoint used to fit to the floor plane, by default 12
    head_id : Optional[int], optional
        Index of the head keypoint used double check that foot keypoints are below head keypoints after rotation, by default None
    dtype : Optional[Type[Union[np.DTypeLike]]], optional
        Desired data type of output array.

    Returns
    -------
    pose : npt.NDArray
        Array of floor-aligned 3D pose values of shape (# frames, # keypoints, 3 coordinates).
    """
    print("Fitting and rotating the floor for each video to alignment ... ")

    # Initial calculation of plane to find outlier values
    [xy, z] = [pose[:, foot_id, :2], pose[:, foot_id, 2]]
    const = np.ones((pose.shape[0], 1))
    coeff = np.linalg.lstsq(np.append(xy, const, axis=1), z, rcond=None)[0]
    z_diff = (
        pose[:, foot_id, 0] * coeff[0] + pose[:, foot_id, 1] * coeff[1] + coeff[2]
    ) - pose[:, foot_id, 2]
    z_mean = np.mean(z_diff)
    z_range = np.std(z_diff) * np.float32(1.5)
    mid_foot_vals = np.where((z_diff > z_mean - z_range) & (z_diff < z_mean + z_range))[
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
    un = np.array([-coeff[0], -coeff[1], 1]) / np.linalg.norm([-coeff[0], -coeff[1], 1])
    vn = np.array([0, 0, 1])
    theta = np.arccos(np.clip(np.dot(un, vn), -1, 1))
    rot_vec = np.cross(un, vn) / np.linalg.norm(np.cross(un, vn)) * theta
    rot_mat = R.from_rotvec(rot_vec).as_matrix().astype(dtype)
    rot_mat = np.expand_dims(rot_mat, axis=2).repeat(
        pose.shape[0] * pose.shape[1], axis=2
    )
    pose[:, :, 2] -= coeff[2]  # Fixing intercept to zero
    # Rotating
    pose_rot = np.einsum("jki,ik->ij", rot_mat, np.reshape(pose, (-1, 3))).reshape(
        pose.shape
    )

    ## Checking to make sure snout is on average above the feet
    if is_whole_body:
        assert np.mean(pose_rot[:, head_id, 2]) > np.mean(
            pose_rot[:, foot_id, 2]
        )  # checking head is above foot

    return pose_rot


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


def z_filter(
    pose: npt.NDArray,
    exp_id: Union[npt.NDArray, List],
    threshold: float = 2500,
    connectivity=None,
):
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


def median_filter(pose: npt.NDArray, ids: npt.ArrayLike, filter_len: int = 5):
    """_summary_

    Parameters
    ----------
    pose : npt.NDArray
        _description_
    ids : Union[npt.NDArray, List]
        _description_
    filter_len : int, optional
        _description_, by default 5

    Returns
    -------
    _type_
        _description_
    """
    print("Applying Median Filter")
    for _, i in enumerate(tqdm(np.unique(ids))):
        pose_exp = pose[ids == i, ...]
        pose[ids == i, ...] = scp_ndi.median_filter(
            pose_exp, (filter_len, 1, 1), mode="nearest"
        )

    return pose


def anipose_med_filt(
    pose: npt.NDArray,
    exp_id: npt.ArrayLike,
    filter_len: int = 6,
    threshold: float = 5,
):
    for _, i in enumerate(tqdm(np.unique(exp_id))):
        pose_exp = pose[exp_id == i, :, :]

        pose_error = pose_exp - scp_ndi.median_filter(
            pose_exp, (filter_len, 1, 1)
        )  # Median filter 5 frames repeat the ends of video
        pose_error = np.linalg.norm(pose_error, axis=-1).mean(axis=-1)

        # plt.hist(pose_error, bins=1000)
        # plt.savefig("../../results/interp_ensemble/err_hist" + str(i) + ".png")
        # plt.close()

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

        pose_error = pose_exp - scp_ndi.median_filter(
            pose_exp, (filter_len, 1, 1)
        )  # Median filter 5 frames repeat the ends of video
        pose_error = np.linalg.norm(pose_error, axis=-1).mean(axis=-1)

        # plt.hist(pose_error, bins=1000)
        # plt.savefig("../../results/interp_ensemble/err_hist_post" + str(i) + ".png")
        # plt.close()

    return pose


def center_spine(pose, keypt_idx=4):
    print("Centering poses to mid spine ...")
    # Center spine_m to (0,0,0)
    return pose - pose[:, keypt_idx : keypt_idx + 1, :]


def rotate_spine(
    pose: npt.NDArray,
    vector: Union[Tuple, npt.ArrayLike] = (4, 3),
    lock_to_x: bool = False,
):
    """Centers mid spine to (0,0,0) and aligns spine_m -> spine_f to x-z plane

    Parameters
    ----------
    pose : npt.NDArray
        Array of 3D pose values of shape (# frames, # keypoints, 3 coordinates). Assumes centered poses.
    vector : Union[Tuple, npt.ArrayLike], optional
        Either a tuple of the indices for (root, forward) keypoints,
        or a precalculated vector per frame, by default (4, 3)
    lock_to_x : bool, optional
        If true, rotate completely to the x-axis (yaw and pitch), by default False

    Returns
    -------
    pose : npt.NDArray
        Rotated array of 3D pose values of shape (# frames, # keypoints, 3 coordinates).
    """

    num_joints = pose.shape[1]
    if len(vector) == 2:
        yaw = -np.arctan2(
            pose[:, vector[1], 1], pose[:, vector[1], 0]
        )  # Find angle to rotate to axis
    else:
        yaw = -np.arctan2(vector[:, 1], vector[:, 0])
        # Find angle to rotate to axis

    if lock_to_x:
        print("Rotating spine to x axis ... ")
        if len(vector) == 2:
            pitch = np.arctan2(pose[:, vector[1], 2], pose[:, vector[1], 0])
        else:
            pitch = np.arctan2(vector[:, 2], vector[:, 0])
    else:
        print("Rotating spine to xz plane ... ")
        pitch = np.zeros(yaw.shape, dtype=pose.dtype)

    # Rotation matrix for pitch and yaw
    rot_mat = np.array(
        [
            [np.cos(yaw) * np.cos(pitch), -np.sin(yaw), np.cos(yaw) * np.sin(pitch)],
            [np.sin(yaw) * np.cos(pitch), np.cos(yaw), np.sin(yaw) * np.sin(pitch)],
            [-np.sin(pitch), np.zeros(len(yaw), dtype=pose.dtype), np.cos(pitch)],
        ]
    ).repeat(num_joints, axis=2)
    pose_rot = np.einsum("jki,ik->ij", rot_mat, np.reshape(pose, (-1, 3))).reshape(
        pose.shape
    )

    return pose_rot