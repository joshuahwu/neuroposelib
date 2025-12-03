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
    pose: npt.NDArray[np.float_],
    foot_id: Optional[int] = 12,
    head_id: Optional[int] = 0,
    dtype: Optional[Type[Union[np.float32, np.float64]]] = np.float32,
) -> npt.NDArray[np.float_]:
    """Due to the camera calibration, predictions may be rotated to different world coordinates.
    Rotate poses per-video so the fitted floor lies on the XY plane.

    This function is a thin wrapper (decorated with [`neuroposelib.utils.by_id`][neuroposelib.utils.by_id]) that
    calls :func:`align_floor` for each video id. It is intended to be applied
    on a full dataset where frames are grouped by id by the decorator.

    Parameters
    ----------
    pose : npt.NDArray
        Array of 3D pose values with shape ``(n_frames, n_keypoints, 3)``.
    foot_id : int, optional
        Index of the keypoint used to fit the floor plane. Default is ``12``.
    head_id : int, optional
        Index of the head keypoint used as a sanity check that the head is
        above the feet after rotation. Default is ``0``.
    dtype : dtype, optional
        Numeric dtype used for rotation matrix computations. Default ``np.float32``.

    Returns
    -------
    npt.NDArray
        Floor-aligned poses with the same shape as ``pose``.

    Raises
    ------
    AssertionError
        If, after rotation, the mean z coordinate of ``head_id`` is not greater
        than the mean z coordinate of ``foot_id`` (sanity check).
    IndexError
        If ``foot_id`` or ``head_id`` are out of range for the provided poses.

    Notes
    -----
    The function relies on :func:`align_floor` to perform the actual
    computations; this wrapper exists so the operation can be applied per-id
    when used with the [`neuroposelib.utils.by_id`][neuroposelib.utils.by_id] decorator.
    """
    return align_floor(pose=pose, foot_id=foot_id, head_id=head_id, dtype=dtype)


def align_floor(
    pose: npt.NDArray[np.float_],
    foot_id: Optional[int] = 12,
    head_id: Optional[int] = None,
    dtype: Optional[npt.DTypeLike] = np.float32,
) -> npt.NDArray[np.float_]:
    """Due to the camera calibration, predictions may be rotated to different world coordinates. 
    Rotate a single-video pose so the floor lies on the XY plane.

    The function fits a plane to the chosen ``foot_id`` keypoint positions
    (using a robust subset to discard outliers), computes a rotation that
    aligns the plane normal to the global Z axis, and applies that rotation to
    the entire pose. The plane intercept is shifted so the plane has z=0.

    Parameters
    ----------
    pose : npt.NDArray
        Array of 3D pose values of shape ``(n_frames, n_keypoints, 3)``.
    foot_id : int, optional
        Index of the foot keypoint used to fit the floor plane. Default ``12``.
    head_id : int, optional
        Index of the head keypoint used for a post-rotation sanity check.
        If ``None``, no check is performed. Default ``None``.
    dtype : dtype, optional
        Numeric dtype for internal rotation matrix. Default ``np.float32``.

    Returns
    -------
    npt.NDArray
        Rotated (floor-aligned) poses with same shape as ``pose``.

    Raises
    ------
    AssertionError
        If ``head_id`` is provided and the mean z of the head is not above the
        mean z of the feet after rotation.
    IndexError
        If ``foot_id`` or ``head_id`` are out of range for the provided poses.

    Notes
    -----
    This implementation uses least-squares plane fitting and removes outlier
    foot samples before recomputing the plane. Numerical degeneracies (e.g.
    when all foot positions are collinear) can produce ``NaN`` or raise
    linear algebra errors. Consider adding small epsilons or additional
    protections if inputs may be ill-conditioned.
    """
    print("Fitting and rotating the floor for each video to alignment ...")

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
    ]

    # Recalculate plane using values with outliers removed
    [xy, z] = [pose[mid_foot_vals, foot_id, :2], pose[mid_foot_vals, foot_id, 2]]
    const = np.ones((xy.shape[0], 1))
    coeff = np.linalg.lstsq(np.append(xy, const, axis=1), z, rcond=None)[0]

    # Compute rotation matrix that aligns fitted normal to global z axis
    un = np.array([-coeff[0], -coeff[1], 1]) / np.linalg.norm([-coeff[0], -coeff[1], 1])
    vn = np.array([0, 0, 1])
    theta = np.arccos(np.clip(np.dot(un, vn), -1, 1))
    rot_vec = np.cross(un, vn) / np.linalg.norm(np.cross(un, vn)) * theta
    rot_mat = R.from_rotvec(rot_vec).as_matrix().astype(dtype)

    # Broadcast rotation matrix and apply to all joints
    rot_mat = np.expand_dims(rot_mat, axis=2).repeat(
        pose.shape[0] * pose.shape[1], axis=2
    )

    # Shift the intercept so floor is at z=0 and rotate
    pose[:, :, 2] -= coeff[2]
    pose_rot = np.einsum("jki,ik->ij", rot_mat, np.reshape(pose, (-1, 3))).reshape(
        pose.shape
    )

    # Sanity check: head should be above feet after rotation (if head_id provided)
    if head_id is not None:
        assert np.mean(pose_rot[:, head_id, 2]) > np.mean(pose_rot[:, foot_id, 2])

    return pose_rot


def vel_filter(
    pose: npt.NDArray[np.float_],
    exp_id: Union[npt.NDArray[np.int_], List[int]],
    threshold: float = 20,
) -> npt.NDArray[np.float_]:
    """Detect and replace high-velocity frames with cubic spline interpolation.

    This function computes a per-frame velocity signal and, for frames that
    exceed ``threshold``, replaces the pose with values interpolated from the
    non-erroneous frames using [`scipy.interpolate.CubicSpline`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.CubicSpline.html).

    Parameters
    ----------
    pose : array-like
        Array of poses with shape ``(n_frames, n_keypoints, 3)``.
    exp_id : array-like
        Array-like of identifiers grouping frames by experiment/video (length
        ``n_frames``). Frames with the same id are processed together.
    threshold : float, optional
        Velocity magnitude threshold above which frames are considered bad.
        Default ``20``.

    Returns
    -------
    npt.NDArray
        The input pose array with high-velocity frames replaced by spline
        interpolated values.

    Raises
    ------
    ValueError
        If interpolation cannot be performed (e.g. all frames are flagged as bad).

    Notes
    -----
    The function uses [`neuroposelib.utils.get_frame_diff`][neuropopselib.utils.get_frame_diff] to compute per-frame differences
    and [`tqdm`](https://tqdm.github.io/) for a progress bar over unique ``exp_id`` values.
    """
    print("Completing cubic spline interpolation based on velocity")
    for _, i in enumerate(tqdm(np.unique(exp_id))):
        pose_exp = pose[exp_id == i, ...]
        dxyz = get_frame_diff(pose_exp, time=1, idx_center=False)
        avg_vel = np.linalg.norm(np.sum(dxyz, axis=-1), axis=-1)

        if np.any(avg_vel > threshold):
            bad_tracking_frames = np.where(avg_vel > threshold)[0]
            good_tracking_frames = np.where(avg_vel <= threshold)[0]

            if len(good_tracking_frames) == 0:
                raise ValueError("No good frames available for spline interpolation")

            cs = CubicSpline(good_tracking_frames, pose_exp[good_tracking_frames, ...])
            pose_exp[bad_tracking_frames, ...] = cs(bad_tracking_frames)

        pose[exp_id == i, ...] = pose_exp
    return pose


def z_filter(
    pose: npt.NDArray[np.float_],
    exp_id: Union[npt.NDArray[np.int_], List[int]],
    threshold: float = 2500,
) -> npt.NDArray[np.float_]:
    """Detect frames with extreme summed-Z and interpolate them.

    For each grouped experiment/video (as indicated by ``exp_id``), the sum
    of the Z-coordinates across keypoints is computed per-frame. Frames where
    that sum exceeds ``threshold`` are replaced via cubic-spline interpolation
    from the remaining frames.

    Parameters
    ----------
    pose : npt.NDArray
        Pose array of shape ``(n_frames, n_keypoints, 3)``.
    exp_id : array-like
        Per-frame experiment/video ids (length ``n_frames``).
    threshold : float, optional
        Sum-of-z threshold above which frames are considered bad. Default ``2500``.

    Returns
    -------
    npt.NDArray
        Pose array with z-outlier frames replaced by interpolated values.

    Raises
    ------
    ValueError
        If interpolation cannot be performed because all frames were flagged as bad.
    """
    print("Completing cubic spline interpolation based on z values")
    for _, i in enumerate(tqdm(np.unique(exp_id))):
        pose_exp = pose[exp_id == i, ...]
        z_trace = np.sum(pose_exp[..., 2], axis=-1)

        if np.any(z_trace > threshold):
            bad_tracking_frames = np.where(z_trace > threshold)[0]
            good_tracking_frames = np.where(z_trace <= threshold)[0]

            if len(good_tracking_frames) == 0:
                raise ValueError("No good frames available for spline interpolation")

            cs = CubicSpline(good_tracking_frames, pose_exp[good_tracking_frames, ...])
            pose_exp[bad_tracking_frames, ...] = cs(bad_tracking_frames)

        pose[exp_id == i, ...] = pose_exp
    return pose


def median_filter(
    pose: npt.NDArray[np.float_],
    ids: Union[npt.NDArray[np.int_], List[int]],
    filter_len: int = 5,
) -> npt.NDArray[np.float_]:
    """Apply a 1D temporal median filter per id/group.

    The median filter is applied along the time axis for each group of frames
    sharing the same id. The spatial and coordinate axes are preserved.

    Parameters
    ----------
    pose : npt.NDArray
        Pose array of shape ``(n_frames, n_keypoints, 3)``.
    ids : array-like
        Array of ids (per frame) indicating grouping for median filtering
        (length ``n_frames``).
    filter_len : int, optional
        Temporal median kernel length (odd integer recommended). Default ``5``.

    Returns
    -------
    npt.NDArray
        Median-filtered pose array with the same shape as the input.
    """
    print("Applying Median Filter")
    for _, i in enumerate(tqdm(np.unique(ids))):
        pose_exp = pose[ids == i, ...]
        pose[ids == i, ...] = scp_ndi.median_filter(
            pose_exp, (filter_len, 1, 1), mode="nearest"
        )
    return pose


def anipose_med_filt(
    pose: npt.NDArray[np.float_],
    exp_id: Union[npt.NDArray[np.int_], List[int]],
    filter_len: int = 6,
    threshold: float = 5,
) -> npt.NDArray[np.float_]:
    """Anipose-style median filtering + error-driven interpolation.

    For each video group, a median filter is used to compute a per-frame error
    signal (how far each frame is from the median-filtered value). Frames
    whose mean error exceeds ``threshold`` are treated as bad and are
    interpolated per-joint and per-axis using cubic splines built from the
    remaining (good) frames.

    Parameters
    ----------
    pose : npt.NDArray
        Pose array of shape ``(n_frames, n_keypoints, 3)``.
    exp_id : array-like
        Per-frame experiment/video ids.
    filter_len : int, optional
        Temporal window size used for the median filter. Default ``6``.
    threshold : float, optional
        Error threshold (mean-over-channels) above which a frame is considered
        bad. Default ``5``.

    Returns
    -------
    npt.NDArray
        Pose array after median-filter cleaning and interpolation.

    Raises
    ------
    ValueError
        If interpolation cannot be performed because all frames were flagged as bad.
    """
    for _, i in enumerate(tqdm(np.unique(exp_id))):
        pose_exp = pose[exp_id == i, :, :]
        pose_error = pose_exp - scp_ndi.median_filter(pose_exp, (filter_len, 1, 1))
        pose_error = np.linalg.norm(pose_error, axis=-1).mean(axis=-1)

        bad_tracking_frames = np.where(pose_error > threshold)[0]
        good_tracking_frames = np.where(pose_error <= threshold)[0]

        if len(good_tracking_frames) == 0:
            raise ValueError("No good frames available for spline interpolation")

        for joint in tqdm(np.arange(pose_exp.shape[1])):
            for ax in np.arange(pose_exp.shape[2]):
                cs = CubicSpline(
                    good_tracking_frames, pose_exp[good_tracking_frames, joint, ax]
                )
                pose_exp[bad_tracking_frames, joint, ax] = cs(bad_tracking_frames)

        pose[exp_id == i, :, :] = pose_exp

    return pose


def center_spine(
    pose: npt.NDArray[np.float_], keypt_idx: int = 4
) -> npt.NDArray[np.float_]:
    """Shift poses so the selected keypoint is at the origin per-frame.

    Parameters
    ----------
    pose : npt.NDArray
        Pose array of shape ``(n_frames, n_keypoints, 3)``.
    keypt_idx : int, optional
        Index of the keypoint to center on (default ``4``).

    Returns
    -------
    npt.NDArray
        Centered pose array with the same shape as ``pose``. The returned
        array equals ``pose - pose[:, keypt_idx:keypt_idx+1, :]``.

    Raises
    ------
    IndexError
        If ``keypt_idx`` is out of range for the keypoints dimension.
    """
    print("Centering poses to mid spine ...")
    return pose - pose[:, keypt_idx : keypt_idx + 1, :]


def rotate_spine(
    pose: npt.NDArray[np.float_],
    vector: Union[Tuple[int, int], npt.NDArray[np.float_]] = (4, 3),
    lock_to_x: bool = False,
) -> npt.NDArray[np.float_]:
    """Rotate poses so the spine-forward direction lies in the XZ plane.

    The function expects poses to be centered (e.g. via [`center_spine`][neuroposelib.preprocess.center]).
    It computes a yaw (and optionally pitch) rotation per-frame that aligns
    the specified forward vector into the XZ plane or locks it to the +X axis.

    Parameters
    ----------
    pose : npt.NDArray
        Centered pose array of shape ``(n_frames, n_keypoints, 3)``.
    vector : tuple or array-like, optional
        Either ``(root_idx, forward_idx)`` indicating which keypoints define
        the forward direction per-frame, or a precomputed per-frame vector
        array of shape ``(n_frames, 3)``. Default ``(4, 3)``.
    lock_to_x : bool, optional
        If ``True``, apply full rotation (yaw + pitch) such that the forward
        direction points to +X. If ``False`` only remove yaw to move the
        forward direction into the XZ plane. Default ``False``.

    Returns
    -------
    npt.NDArray
        Rotated pose array with the same shape as input.

    Raises
    ------
    IndexError
        If provided keypoint indices are out of range.
    ValueError
        If the provided ``vector`` has an unexpected shape.
    """
    num_joints = pose.shape[1]

    # Finding yaw angle
    if len(vector) == 2:
        yaw = -np.arctan2(pose[:, vector[1], 1], pose[:, vector[1], 0])
    else:
        yaw = -np.arctan2(vector[:, 1], vector[:, 0])

    if lock_to_x:
        print("Rotating spine to x axis ... ")
        if len(vector) == 2:
            pitch = np.arctan2(pose[:, vector[1], 2], pose[:, vector[1], 0])
        else:
            pitch = np.arctan2(vector[:, 2], vector[:, 0])
    else:
        print("Rotating spine to xz plane ...")
        pitch = np.zeros(yaw.shape, dtype=pose.dtype)

    # Construct per-frame rotation matrices (yaw then pitch)
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


def qnormalize(q: npt.NDArray[np.float_]) -> npt.NDArray[np.float_]:
    norm = np.linalg.norm(q, axis=-1, keepdims=True)
    return q / norm


def qbetween(
    v0: npt.NDArray[np.float_], v1: npt.NDArray[np.float_]
) -> npt.NDArray[np.float_]:
    """Compute quaternion(s) that rotate ``v0`` to ``v1``.

    Parameters
    ----------
    v0 : np.ndarray
        Source vector(s) of shape ``(..., 3)``.
    v1 : np.ndarray
        Target vector(s) of shape ``(..., 3)``.

    Returns
    -------
    np.ndarray
        Unit quaternion(s) in ``(w, x, y, z)`` order with shape ``(..., 4)``.

    Raises
    ------
    AssertionError
        If the last dimension of ``v0`` or ``v1`` is not 3.
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
    """Return the conjugate (inverse for unit quaternions) of ``q``.

    Parameters
    ----------
    q : np.ndarray
        Quaternion(s) in ``(w, x, y, z)`` order with shape ``(..., 4)``.

    Returns
    -------
    np.ndarray
        Conjugated quaternion(s) with same shape as ``q``.

    Raises
    ------
    AssertionError
        If the last dimension of ``q`` is not 4.
    """
    assert q.shape[-1] == 4, "q must be an array of shape (*, 4)"
    mask = np.ones_like(q)
    mask[..., 1:] = -1
    return q * mask


def qmul(
    q: npt.NDArray[np.float_], r: npt.NDArray[np.float_]
) -> npt.NDArray[np.float_]:
    """Multiply quaternions: compute ``r * q`` for inputs in ``(w,x,y,z)``.

    Parameters
    ----------
    q, r : np.ndarray
        Input quaternion arrays with last dimension ``4``. Shapes must be
        broadcastable.

    Returns
    -------
    np.ndarray
        Quaternion product with the broadcasted shape and last dimension ``4``.

    Raises
    ------
    AssertionError
        If the last dimension of either input is not 4.
    """
    assert q.shape[-1] == 4
    assert r.shape[-1] == 4
    q = q.reshape(-1, 4)
    r = r.reshape(-1, 4)
    w = r[:, 0] * q[:, 0] - r[:, 1] * q[:, 1] - r[:, 2] * q[:, 2] - r[:, 3] * q[:, 3]
    x = r[:, 0] * q[:, 1] + r[:, 1] * q[:, 0] - r[:, 2] * q[:, 3] + r[:, 3] * q[:, 2]
    y = r[:, 0] * q[:, 2] + r[:, 1] * q[:, 3] + r[:, 2] * q[:, 0] - r[:, 3] * q[:, 1]
    z = r[:, 0] * q[:, 3] - r[:, 1] * q[:, 2] + r[:, 2] * q[:, 1] + r[:, 3] * q[:, 0]
    result = np.stack((w, x, y, z), axis=1)
    return result.reshape(q.shape)


def inv_kin(
    pose: npt.NDArray[np.float_],
    kinematic_tree: Union[List, np.ndarray],
    offset: npt.NDArray[np.float_],
    forward_indices: Union[List[int], npt.NDArray[np.int_]] = [0, 1],
) -> npt.NDArray[np.float_]:
    """Compute local joint quaternions from global joint positions.

    Adapted from the T2M-GPT implementation, this function computes an
    inverse-kinematics-like local quaternion representation for each joint.
    For each bone defined in ``kinematic_tree`` the function computes the
    quaternion that rotates the rest offset to the observed bone direction.

    Parameters
    ----------
    pose : np.ndarray
        Global 3D joint positions with shape ``(n_frames, n_joints, 3)``.
    kinematic_tree : list or np.ndarray
        Iterable of kinematic chains; each chain is an iterable of joint
        indices in parent->child order.
    offset : np.ndarray
        Rest offsets (bone vectors) indexed by joint index, typically with
        shape ``(n_joints, 3)``.
    forward_indices : list-like, optional
        Pair of joint indices used to determine the forward direction for the
        root rotation. Default ``[0, 1]``.

    Returns
    -------
    np.ndarray
        Local quaternion representation for each joint with shape
        ``(n_frames, n_joints, 4)``. ``local_quat[:, j, :]`` contains the
        quaternion for joint ``j``.

    Raises
    ------
    IndexError
        If indices in ``kinematic_tree`` or ``forward_indices`` are invalid.
    ValueError
        If ``offset`` and ``pose`` shapes are inconsistent.

    Notes
    -----
    This routine assumes non-zero bone lengths and non-degenerate forward
    vectors. It does not add numeric epsilons when normalizing vectors; if
    inputs may contain zero-length bones consider adding safeguards.
    """
    # Find forward root direction
    forward = pose[:, forward_indices[1], :] - pose[:, forward_indices[0], :]
    forward = forward / np.linalg.norm(forward, axis=-1)[..., None]

    # Root rotation that maps forward -> +X
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
    continuous_6D: npt.NDArray[np.float_],
    kinematic_tree: Union[List, np.ndarray],
    offset: npt.NDArray[np.float_],
    root_pos: npt.NDArray[np.float_],
    do_root_R: bool = True,
) -> npt.NDArray[np.float_]:
    """Forward kinematics using 6D continuous rotation representations.

    Convert per-joint 6D continuous rotations into rotation matrices, multiply
    along kinematic chains and apply offsets to compute global joint
    positions.

    Parameters
    ----------
    continuous_6D : np.ndarray
        Array of continuous 6D rotations with shape ``(batch_size, n_joints, 6)``.
    kinematic_tree : list or np.ndarray
        Kinematic chains as used by :func:`inv_kin`.
    offset : np.ndarray
        Rest offsets per joint (``(n_joints, 3)``) or batched offsets.
    root_pos : np.ndarray
        Root joint positions for each batch, shape ``(batch_size, 3)``.
    do_root_R : bool, optional
        If ``True`` apply the root rotation from ``continuous_6D[:, 0]``.

    Returns
    -------
    np.ndarray
        Global joint positions with shape ``(batch_size, n_joints, 3)``.
    """
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
            pose[:, chain[i]] = (
                np.matmul(matR, offset_vec).squeeze(-1) + pose[:, chain[i - 1]]
            )

    return pose


def cont6d_to_matrix(
    cont6d: npt.NDArray[np.float_], eps: float = 0
) -> npt.NDArray[np.float_]:
    """Convert a 6D continuous rotation representation to 3x3 matrices.

    The 6D representation stores two 3D vectors; the implementation uses a
    Gram–Schmidt-like orthonormalization to construct a valid rotation matrix.

    Parameters
    ----------
    cont6d : np.ndarray
        Array with last-dimension ``6`` representing the continuous rotation.
    eps : float, optional
        Small epsilon added to denominators to avoid division by zero. Default
        ``0`` (caller should set a small positive value if inputs may contain
        zeros).

    Returns
    -------
    npt.NDArray
        Rotation matrices with shape ``(..., 3, 3)``.

    Raises
    ------
    AssertionError
        If ``cont6d`` does not have ``6`` as the last dimension.
    """
    assert cont6d.shape[-1] == 6, "The last dimension must be 6"
    x_raw = cont6d[..., 0:3]
    y_raw = cont6d[..., 3:6]
    x = x_raw / (np.linalg.norm(x_raw, axis=-1, keepdims=True) + eps)
    z = np.cross(x, y_raw, axis=-1)
    z = z / (np.linalg.norm(z, axis=-1, keepdims=True) + eps)
    y = np.cross(z, x, axis=-1)

    x = np.expand_dims(x, axis=-1)
    y = np.expand_dims(y, axis=-1)
    z = np.expand_dims(z, axis=-1)
    mat = np.concatenate([x, y, z], axis=-1)
    return mat
