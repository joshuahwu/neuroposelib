import scipy.ndimage as scp_ndi
from scipy.interpolate import CubicSpline
import numpy as np
import numpy.typing as npt
from neuroposelib.utils import by_id, get_frame_diff
from typing import Optional, Union, List, Type, Tuple
from tqdm import tqdm
from scipy.spatial.transform import Rotation as R

@by_id
def align_floor_by_id(
    pose: npt.NDArray,
    foot_id: Optional[int] = 12,
    head_id: Optional[int] = 0,
    dtype: Optional[Type[Union[np.float32, np.float64]]] = np.float32,
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


def align_floor(
    pose: npt.NDArray,
    foot_id: Optional[int] = 12,
    head_id: Optional[int] = None,
    dtype: Optional[npt.DTypeLike] = np.float32,
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

## The following code is adapted from 

def qnormalize(q):
    norm = np.linalg.norm(q, axis=-1, keepdims=True)
    return q / norm

def qbetween(v0: npt.NDArray[np.float_], v1: npt.NDArray[np.float_]) -> npt.NDArray[np.float_]:
    """
    Compute the quaternion that rotates vector v0 to vector v1.

    Parameters
    ----------
    v0 : np.ndarray of shape (..., 3)
        The source vector(s).
    v1 : np.ndarray of shape (..., 3)
        The target vector(s).

    Returns
    -------
    np.ndarray of shape (..., 4)
        Unit quaternion(s) representing the rotation from v0 to v1, in (w, x, y, z) format.
    """
    assert v0.shape[-1] == 3, "v0 must be of the shape (*, 3)"
    assert v1.shape[-1] == 3, "v1 must be of the shape (*, 3)"

    v = np.cross(v0, v1, axis=-1)
    w = np.sqrt(
        np.sum(v0**2, axis=-1, keepdims=True) * np.sum(v1**2, axis=-1, keepdims=True)
    ) + np.sum(v0 * v1, axis=-1, keepdims=True)
    q = np.concatenate([w, v], axis=-1)
    return qnormalize(q)

def qinv(q: npt.NDArray[np.float_]) -> npt.NDArray[np.float_]:
    """
    Compute the inverse (conjugate) of a unit quaternion.

    Parameters
    ----------
    q : np.ndarray of shape (..., 4)
        Input quaternion(s) in (w, x, y, z) format.

    Returns
    -------
    np.ndarray of shape (..., 4)
        The conjugate of the input quaternion(s), which is the inverse if the input is unit-norm.
    """
    assert q.shape[-1] == 4, "q must be an array of shape (*, 4)"
    mask = np.ones_like(q)
    mask[..., 1:] = -1
    return q * mask

def qmul(q: npt.NDArray[np.float_], r: npt.NDArray[np.float_]) -> npt.NDArray[np.float_]:
    """
    Multiply quaternion(s) q with quaternion(s) r.

    Parameters
    ----------
    q : np.ndarray of shape (..., 4)
        The left quaternion(s) in (w, x, y, z) format.
    r : np.ndarray of shape (..., 4)
        The right quaternion(s) in (w, x, y, z) format.

    Returns
    -------
    np.ndarray of shape (..., 4)
        The product quaternion(s) representing r * q.
    """
    assert q.shape[-1] == 4
    assert r.shape[-1] == 4

    q = q.reshape(-1, 4)
    r = r.reshape(-1, 4)

    w = r[:, 0]*q[:, 0] - r[:, 1]*q[:, 1] - r[:, 2]*q[:, 2] - r[:, 3]*q[:, 3]
    x = r[:, 0]*q[:, 1] + r[:, 1]*q[:, 0] - r[:, 2]*q[:, 3] + r[:, 3]*q[:, 2]
    y = r[:, 0]*q[:, 2] + r[:, 1]*q[:, 3] + r[:, 2]*q[:, 0] - r[:, 3]*q[:, 1]
    z = r[:, 0]*q[:, 3] - r[:, 1]*q[:, 2] + r[:, 2]*q[:, 1] + r[:, 3]*q[:, 0]

    result = np.stack((w, x, y, z), axis=1)
    return result.reshape(q.shape)

def inv_kin(
    pose: np.ndarray,
    kinematic_tree: Union[List, np.ndarray],
    offset: np.ndarray,
    forward_indices: Union[List, np.ndarray] = [0, 1],
):
    """
    Adapted from T2M-GPT (https://mael-zys.github.io/T2M-GPT/)
    [1] Zhang, Jianrong, et al. "Generating Human Motion From Textual
    Descriptions With Discrete Representations." Proceedings of the
    IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2023.
    """

    # Find forward root direction
    forward = pose[:, forward_indices[1], :] - pose[:, forward_indices[0], :]
    forward = forward / np.linalg.norm(forward, axis=-1)[..., None]

    # Root Rotation
    target = np.array([[1, 0, 0]]).repeat(len(forward), axis=0)
    root_quat = qbetween(forward, target)

    local_quat = np.zeros(pose.shape[:-1] + (4,))
    root_quat[0] = np.array([[1.0, 0.0, 0.0, 0.0]])
    local_quat[:, 0] = root_quat
    for chain in kinematic_tree:
        R = root_quat
        for i in range(len(chain) - 1):
            u = offset[chain[i + 1]][None, ...].repeat(len(pose), axis=0)
            v = pose[:, chain[i + 1]] - pose[:, chain[i]]
            v = v / np.linalg.norm(v, axis=-1)[..., None]
            rot_u_v = qbetween(u, v)
            R_loc = qmul(qinv(R), rot_u_v)
            local_quat[:, chain[i + 1], :] = R_loc
            R = qmul(R, R_loc)

    return local_quat


def fwd_kin_cont6d(
    continuous_6D: np.ndarray,
    kinematic_tree: Union[List, np.ndarray],
    offset: np.ndarray,
    root_pos: np.ndarray,
    do_root_R: bool = True,
):
    # continuous_6D (batch_size, pose_num, 6)
    # pose (batch_size, pose_num, 3)
    # root_pos (batch_size, 3)

    pose = np.zeros(continuous_6D.shape[:-1] + (3,))
    pose[:, 0] = root_pos

    if len(offset.shape) == 2:
        offsets = np.moveaxis(np.tile(offset[..., None], continuous_6D.shape[0]), -1, 0)
    else:
        offsets = offset

    for chain in kinematic_tree:
        if do_root_R:
            matR = cont6d_to_matrix(continuous_6D[:, 0])
        else:
            matR = np.eye(3)[np.newaxis, :].repeat(len(continuous_6D), axis=0)
        for i in range(1, len(chain)):
            matR = np.matmul(matR, cont6d_to_matrix(continuous_6D[:, chain[i]]))
            offset_vec = offsets[:, chain[i]][..., np.newaxis]
            # print(matR.shape, offset_vec.shape)
            pose[:, chain[i]] = (
                np.matmul(matR, offset_vec).squeeze(-1) + pose[:, chain[i - 1]]
            )
    return pose

def cont6d_to_matrix(cont6d: npt.NDArray[np.float_], eps: float = 0) -> npt.NDArray[np.float_]:
    """
    Convert 6D continuous rotation representation to a 3x3 rotation matrix.

    Parameters
    ----------
    cont6d : np.ndarray of shape (..., 6)
        The 6D rotation representation. First 3 elements define the x-axis direction,
        next 3 are auxiliary vector for y-axis construction.
    eps : float, optional
        Small epsilon to prevent division by zero during normalization.

    Returns
    -------
    np.ndarray of shape (..., 3, 3)
        The resulting 3x3 rotation matrices.
    """
    assert cont6d.shape[-1] == 6, "The last dimension must be 6"

    x_raw = cont6d[..., 0:3]
    y_raw = cont6d[..., 3:6]

    x = x_raw / (np.linalg.norm(x_raw, axis=-1, keepdims=True) + eps)
    z = np.cross(x, y_raw, axis=-1)
    z = z / (np.linalg.norm(z, axis=-1, keepdims=True) + eps)
    y = np.cross(z, x, axis=-1)

    # Expand dims to concatenate along last axis
    x = np.expand_dims(x, axis=-1)
    y = np.expand_dims(y, axis=-1)
    z = np.expand_dims(z, axis=-1)

    mat = np.concatenate([x, y, z], axis=-1)
    return mat
