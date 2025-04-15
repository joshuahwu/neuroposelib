from features import *
import DataStruct as ds
import visualization as vis
import interface as itf
import numpy as np
import time
import read, write
from embed import Watershed, Embed
import pickle
import analysis

analysis_key = "ensemble_16"
paths = read.config("../configs/path_configs/" + analysis_key + ".yaml")
params = read.config("../configs/param_configs/fitsne.yaml")
connectivity = read.connectivity(
    path=paths["skeleton_path"], skeleton_name=paths["skeleton_name"]
)

# pose, id, meta, meta_by_frame = read.pose_from_meta(
#     path=paths["meta_path"], connectivity=connectivity
# )

pose = read.pose_mat(paths["pose_path"], connectivity)

vid_id = read.ids(paths["pose_path"], paths["exp_key"])

meta, meta_by_frame = read.meta(paths["meta_path"], id=vid_id)

data_obj = ds.DataStruct(
    pose=pose,
    id=vid_id,
    id_full=vid_id,
    meta=meta,
    meta_by_frame=meta_by_frame,
    connectivity=connectivity,
)

# Separate videos have rotated floor planes - this rotates them back
data_obj.pose = align_floor(pose, vid_id)

# Checking velocity before filtering
avg_vel = np.mean(
    np.linalg.norm(
        get_frame_diff(pose[vid_id == 0, ...], 1, idx_center=False), axis=-1
    ),
    axis=-1,
)
high_vel_frame = np.argmax(avg_vel)
vis.feature_hist(avg_vel, "avg_vel_prefilter", paths["out_path"])
vis.skeleton_vid3D(
    pose[vid_id == 0, ...],
    # avg_vel,
    connectivity,
    frames=[high_vel_frame],
    N_FRAMES=100,
    fps=90,
    dpi=100,
    VID_NAME="high_vel_prefilter_new.mp4",
    SAVE_ROOT=paths["out_path"],
)

# Cleaning up some bad tracking frames
# pose = median_filter(pose,vid_id,filter_len=5) # Regular median filter
pose = z_filter(pose, vid_id, threshold=2000)  # , connectivity = connectivity)
# pose = vel_filter(pose,vid_id,threshold=100)#,max_iter=10, connectivity = connectivity) # Finds location of high velocity, removes, and interpolates value

# Checking velocity after filtering
avg_vel = np.mean(
    np.linalg.norm(
        get_frame_diff(pose[vid_id == 0, ...], 1, idx_center=False), axis=-1
    ),
    axis=-1,
)
vis.feature_hist(avg_vel, "avg_vel_postfilter", paths["out_path"])
vis.skeleton_vid3D(
    pose[vid_id == 0, ...],
    # avg_vel,
    connectivity,
    frames=[high_vel_frame],
    N_FRAMES=100,
    fps=90,
    dpi=100,
    VID_NAME="high_vel_postfilter_new.mp4",
    SAVE_ROOT=paths["out_path"],
)

vis.skeleton_vid3D(
    pose[vid_id == 0, ...],
    # avg_vel,
    connectivity,
    frames=[np.argmax(avg_vel)],
    N_FRAMES=100,
    fps=90,
    dpi=100,
    VID_NAME="high_new_vel_postfilter_new.mp4",
    SAVE_ROOT=paths["out_path"],
)
