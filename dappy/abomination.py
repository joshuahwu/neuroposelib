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

analysis_key = "embedding_analysis_ws_r01"
paths = read.config("../configs/path_configs/" + analysis_key + ".yaml")
params = read.config("../configs/param_configs/fitsne.yaml")
connectivity = read.connectivity(
    path=paths["skeleton_path"], skeleton_name=paths["skeleton_name"]
)

# pose= read.pose_mat(paths['pose_path'],
#                 connectivity)

# vid_id = read.ids(paths['pose_path'], key=paths['exp_key'])

# meta, meta_by_frame = read.meta(paths['meta_path'],
#                                 id = vid_id)

# # Separate videos have rotated floor planes - this rotates them back
# pose = align_floor(pose, vid_id)
# # pose_orig = align_floor_by_id(pose, vid_id)
# write.pose_h5(pose,vid_id,paths['data_path'] + 'pose_aligned32.h5')

pose, vid_id = read.pose_h5(paths['data_path'] + 'pose_aligned32.h5')

pose = median_filter(pose,vid_id,filter_len=5) # Regular median filter

# vis.skeleton_vid3D(pose,
#                    connectivity,
#                    frames=[2000],
#                    N_FRAMES = 150,
#                    dpi=100,
#                    VID_NAME = 'original.mp4',
#                    SAVE_ROOT = paths['out_path'])

abomination_legs = (pose[:,6:12,:]+pose[:,12:,:])/2

abomination_pose = np.append(pose[:,:6,:],abomination_legs,axis=1)

abomination_connectivity = read.connectivity(
    path=paths["skeleton_path"], skeleton_name="mouse_abomination"
)


vis.skeleton_vid3D(abomination_pose,
                   abomination_connectivity,
                   frames=[2000],
                   N_FRAMES = 150,
                   dpi=100,
                   VID_NAME = 'abomination.mp4',
                   SAVE_ROOT = paths['out_path'])