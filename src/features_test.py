from features import *
import DataStruct as ds
import visualization as vis
import numpy as np
import pickle

pose_struct = ds.DataStruct(config_path = '../configs/path_configs/embedding_analysis_dcc_r01.yaml')
# pose_struct.load_connectivity()
# pose_struct.load_pose()
# pose_struct.load_feats()

# # Separate videos have rotated floor planes - this rotates them back
# pose_aligned = align_floor(pose_struct.pose_3d, 
#                            pose_struct.exp_ids_full)
# # Check alignment
# # vis.skeleton_vid3D(pose,
# #                    pose_struct.connectivity,
# #                    frames=[1e3,5e5,2e6],
# #                    N_FRAMES = 200,
# #                    VID_NAME='vid_rotated.mp4',
# #                    SAVE_ROOT='./')

# # Calculating velocities and standard deviation of velocites over windows
# abs_vel = get_velocities(pose_aligned, 
#                          pose_struct.exp_ids_full, 
#                          pose_struct.connectivity.joint_names,
#                          joints=[0,4,5])

# pose = center_spine(pose_aligned)

# rel_vel = get_velocities(pose,
#                          pose_struct.exp_ids_full, 
#                          pose_struct.connectivity.joint_names,
#                          joints=np.delete(np.arange(18),4))

# pose = rotate_spine(pose)
# # vis.skeleton_vid3D(pose,
# #                    pose_struct.connectivity,
# #                    frames=[1000],
# #                    N_FRAMES = 300,
# #                    VID_NAME='vid_centered.mp4',
# #                    SAVE_ROOT='./')

# euclid_vec = get_ego_pose(pose,
#                           pose_struct.connectivity.joint_names)

# angles = get_angles(pose,
#                     pose_struct.connectivity.angles)

# ang_vel = get_angular_vel(angles,
#                           pose_struct.exp_ids_full)


# features = pd.concat([abs_vel, rel_vel, euclid_vec, angles, ang_vel], axis=1)
# del pose, abs_vel, rel_vel, euclid_vec, angles, ang_vel

# pickle.dump(features,open(''.join([pose_struct.out_path,'features.p']), "wb"),protocol=4)

features = pickle.load(open(''.join([pose_struct.out_path,'features.p']),"rb"))
import time
import psutil
print(psutil.vertiual_memory())

methods = ['fbpca']

for i, m in enumerate(methods):
    t = time.time()
    pca_feats = pca(features,
                    keys = ['vel','ego_euc','ang','avel'],
                    n_pcs = 10,
                    method = m)
    print("Time " + m)
    print(time.time() - t)

# feats = np.concatenate((euc_vec, angles, abs_vel, head_angular),axis=1)

import pdb; pdb.set_trace()
#mean center before pca, separate for

# w_let = wavelet(feats_pca)

import pdb; pdb.set_trace()