# from features import *
# import DataStruct as ds
# import visualization as vis
# import interface as itf
import numpy as np
# import time
import read, write
# from embed import Watershed, Embed
# import pickle
# import analysis
# import run
# from pathlib import Path
# from tqdm import tqdm

analysis_key = "ntu_human"
paths = read.config("../configs/path_configs/" + analysis_key + ".yaml")
params = read.config("../configs/param_configs/fitsne.yaml")

connectivity = read.connectivity(
        path=paths["skeleton_path"], skeleton_name=paths["skeleton_name"]
    )

data_dir = paths['data_path']
labels = np.load(data_dir + 'labels.npy', allow_pickle=True)
meta = np.load(data_dir + 'meta.npy', allow_pickle=True)
pose = np.load(data_dir + 'data.npy', allow_pickle=True)

body_idx = [0,3,12,15,13,16,18,20,14,17,19,21,1,4,7,10,2,5,8,11,9,24,36,39,51]



not_test = meta[:,1]<2
labels = labels[not_test]
pose = pose[not_test]
meta = meta[not_test]

num_vids = len(labels)

for i in range(num_vids):
    curr_vid = labels[i]
    # if len(curr_vid)==1:
    #     print(curr_vid)
    if isinstance(curr_vid[0],str) or len(curr_vid)==1:
        continue
        # mid_frame = int(curr_pose.shape[0]/2)
        # vis.skeleton_vid3D(curr_pose*1000, connectivity,
        #                    [mid_frame],
        #                    N_FRAMES = mid_frame*2,
        #                    dpi=100,
        #                    VID_NAME = 'test.mp4',
        #                    SAVE_ROOT = paths['out_path'])
    else:
        action_labels = list(zip(*labels[i]))[0]
        curr_pose = pose[i][:,body_idx,:][...,[0,2,1]]
        import pdb; pdb.set_trace()

# extract_column = lambda column: [list(zip(*labels[i]))[column] for i in range(num_vids)]
# action_labels = extract_column(0)
# action_start = extract_column(1)
# action_end = extract_column(2)

# import pdb; pdb.set_trace()