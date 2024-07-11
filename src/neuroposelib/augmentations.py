import numpy as np
import pandas as pd
from typing import Union, List, Dict
import numpy as np
import tqdm
from scipy.interpolate import CubicSpline


def expand_meta(meta: pd.DataFrame, ids: pd.DataFrame, reps: int):
    # Tile pandas DataFrame
    meta_expanded = pd.DataFrame(np.tile(meta.values.T, reps).T, columns=meta.columns)
    meta_expanded["old_id"] = meta_expanded["id"].copy()
    meta_expanded["id"] = meta_expanded.index

    # Repeating ids
    # ids_expanded = np.tile(ids, reps).reshape((reps, -1))
    # ids_expanded += np.arange(reps)[:, None] * (np.max(ids_expanded) + 1)
    ids_expanded = (
        np.add.outer(ids, np.arange(reps) * (np.max(ids) + 1))
        .T.flatten()
        .astype(ids.dtype)
    )

    assert (
        np.sum(
            ids_expanded[-len(ids) :]
            - ids_expanded[: len(ids)]
            - (np.max(ids) + 1) * (reps - 1)
        )
        == 0
    )
    assert np.sum(ids_expanded[: len(ids)] - ids) == 0

    return meta_expanded, ids_expanded


def joint_ablation(
    pose: np.ndarray,
    joint_groups: Dict,
    ablations: List,
    ids: Union[np.ndarray, List],
    meta: pd.DataFrame,
):
    # Pose is forward facing and centered
    if any([" " in key for key in joint_groups.keys()]):
        raise Exception("Cannot be spaces in joint_groups keys")

    pose_means = pose.mean(axis=0)
    label = ["Normal"] * len(meta)
    pose_augmented = [pose]
    ## Ablations
    for i, keys in enumerate(tqdm.tqdm(ablations)):
        if " " in keys:  # For multi-group ablation
            keypt_indices = []
            for key in keys.split(" "):
                keypt_indices += joint_groups[key]
        else:
            keypt_indices = joint_groups[keys]
        pose_temp = pose.copy()
        # Impute with the mean
        pose_temp[:, keypt_indices, :] = pose_means[None, keypt_indices, :]
        pose_augmented += [pose_temp]

        ## Adjust metadata
        label += [keys] * len(meta)

    pose_augmented = np.concatenate(pose_augmented, axis=0)

    n_augs = len(ablations) + 1
    meta_expanded, ids_expanded = expand_meta(meta, ids, n_augs)
    meta_expanded["joint_ablation"] = label

    return pose_augmented, meta_expanded, ids_expanded


def mirror(
    pose: np.ndarray,
    joint_pairs: np.ndarray,
    ids: Union[np.ndarray, List],
    meta: pd.DataFrame,
):
    # Pose is forward facing and centered
    pose_flip = pose.copy()
    pose_flip[..., 1] = -pose_flip[..., 1]  # Flip Y
    joint_temp = pose_flip[:, joint_pairs[:, 0], :]
    pose_flip[:, joint_pairs[:, 0], :] = pose_flip[
        :, joint_pairs[:, 1], :
    ]  # Flip joint pairs
    pose_flip[:, joint_pairs[:, 1], :] = joint_temp
    pose_augmented = np.concatenate([pose, pose_flip], axis=0)

    meta_expanded, ids_expanded = expand_meta(meta, ids, 2)
    label = ["Original"] * len(meta) + ["Mirrored"] * len(meta)
    meta_expanded["mirror"] = label

    return pose_augmented, meta_expanded, ids_expanded


def noise(
    pose: np.ndarray,
    level: Union[np.ndarray, List],
    ids: Union[np.ndarray, List],
    meta: pd.DataFrame,
):
    print("Augmenting poses with noise")
    # Assumes first element of "level" is 0
    noise = np.zeros(np.shape(pose), dtype=pose.dtype)
    for lev in level[1:]:
        rng = np.random.default_rng()
        noise = np.append(
            noise, np.random.normal(0, lev, np.shape(pose)), axis=0
        ).astype(pose.dtype)

    pose_augmented = np.tile(pose.T, len(level)).T
    assert np.sum(pose_augmented[: len(pose)] - pose) == 0
    pose_augmented += noise
    meta_expanded, ids_expanded = expand_meta(meta, ids, len(level))
    meta_expanded["noise"] = np.repeat(level, np.max(ids) + 1)

    return pose_augmented, meta_expanded, ids_expanded


def speed(
    pose: np.ndarray,
    level: Union[np.ndarray, List],
    ids: Union[np.ndarray, List],
    meta: pd.DataFrame,
):
    pose_augmented, ids_expanded = [], []
    for i, id in enumerate(tqdm.tqdm(np.unique(ids))):
        n_frames = np.sum(ids == id)
        # Fit interpolator to
        cs = CubicSpline(np.arange(n_frames), pose[ids == id, ...])
        for j, lvl in enumerate(level):
            new_t = np.linspace(0, n_frames - 1, int(n_frames / lvl))
            pose_interpolated = cs(new_t)

            pose_augmented += [pose_interpolated]
            curr_id = i * len(level) + j
            ids_expanded += [np.ones(pose_interpolated.shape[0]) * curr_id]

    pose_augmented = np.concatenate(pose_augmented, axis=0)
    ids_expanded = np.concatenate(ids_expanded)

    # For speed unlike w/the other augmentations, we repeat instead of tile
    meta_expanded = pd.DataFrame(
        np.repeat(meta.values, len(level), axis=0), columns=meta.columns
    )
    meta_expanded["old_id"] = meta_expanded["id"].copy()
    meta_expanded["id"] = meta_expanded.index
    meta_expanded["speed"] = np.tile(level, np.max(ids) + 1)

    return pose_augmented, meta_expanded, ids_expanded


def pitch(
    pose: np.ndarray,
    ids: Union[np.ndarray, List],
    meta: pd.DataFrame,
    n_levels: int = 10,
    spinef_idx: int = 3,
):
    pose_augmented = []
    pitch_level = np.linspace(0, 1, n_levels)  # Adjusts pitch by fraction of 1
    pitch = np.arctan2(pose[:, spinef_idx, 2], pose[:, spinef_idx, 0])
    n_joints = pose.shape[1]

    for i, level in enumerate(tqdm.tqdm(pitch_level)):
        pitch_adjusted = pitch * level
        rot_mat = np.array(
            [
                [np.cos(pitch_adjusted), np.zeros(len(pitch)), np.sin(pitch_adjusted)],
                [np.zeros(len(pitch)), np.ones(len(pitch)), np.zeros(len(pitch))],
                [-np.sin(pitch_adjusted), np.zeros(len(pitch)), np.cos(pitch_adjusted)],
            ]
        ).repeat(n_joints, axis=2)

        pose_rot = np.einsum("jki,ik->ij", rot_mat, np.reshape(pose, (-1, 3))).reshape(
            pose.shape
        )

        pose_augmented += [pose_rot]

    pose_augmented = np.concatenate(pose_augmented, axis=0)
    meta_expanded, ids_expanded = expand_meta(meta, ids, n_levels)
    meta_expanded["pitch"] = np.repeat(pitch_level, np.max(ids) + 1)
    return pose_augmented, meta_expanded, ids_expanded
