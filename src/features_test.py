from features import *
import DataStruct as ds
import visualization as vis
import interface as itf
import numpy as np
import time

# t = time.time()
# pstruct = ds.DataStruct(config_path = '../configs/embedding_analysis_ws_r01.yaml')
# pstruct.load_connectivity()
# pstruct.load_pose()

# # Separate videos have rotated floor planes - this rotates them back
# # Also contains a median filter
# pstruct.pose_3d = align_floor(pstruct.pose_3d, 
#                               pstruct.exp_ids_full)

# # # Calculating velocities and standard deviation of velocites over windows
# # abs_vel, abs_vel_labels  = get_velocities(pstruct.pose_3d, 
# #                                           pstruct.exp_ids_full, 
# #                                           pstruct.connectivity.joint_names,
# #                                           joints=[0,4,5])

# # # Centering all joint locations to mid-spine
# # pose = center_spine(pstruct.pose_3d)

# # # Getting relative velocities
# # rel_vel, rel_vel_labels = get_velocities(pose,
# #                                          pstruct.exp_ids_full, 
# #                                          pstruct.connectivity.joint_names,
# #                                          joints=np.delete(np.arange(18),4))

# # # Rotates front spine to xz axis
# # pose = rotate_spine(pose)

# # # Reshape pose to get egocentric pose features
# # ego_pose, ego_pose_labels  = get_ego_pose(pose,
# #                                           pstruct.connectivity.joint_names)

# # # Calculating joint angles
# # angles, angle_labels = get_angles(pose,
# #                                   pstruct.connectivity.angles)

# # # Calculating angle velocities
# # ang_vel, ang_vel_labels = get_angular_vel(angles,
# #                                           angle_labels,
# #                                           pstruct.exp_ids_full)

# # # Collect all features together
# # features = np.concatenate([abs_vel, rel_vel, ego_pose, angles, ang_vel], axis=1)
# # labels = abs_vel_labels + rel_vel_labels + ego_pose_labels + angle_labels + ang_vel_labels

# # # Clear memory
# # del pose, abs_vel, rel_vel, ego_pose, angles, ang_vel
# # del abs_vel_labels, rel_vel_labels, ego_pose_labels, angle_labels, ang_vel_labels

# # # Save postural features to h5 file
# # save_h5(features, labels, path = ''.join([pstruct.out_path,'postural_feats.h5']))

# # Read postural features from h5 file
# features, labels = read_h5(path = ''.join([pstruct.out_path,'postural_feats.h5']))

# pc_feats, pc_labels = pca(features,
#                             labels,
#                             categories = ['vel','ego_euc','ang','avel'],
#                             n_pcs = 8,
#                             method = 'fbpca')

# del features, labels

# wlet_feats, wlet_labels = wavelet(pc_feats, 
#                                   pc_labels,
#                                   pstruct.exp_ids_full)

# save_h5(wlet_feats, wlet_labels, path = ''.join([pstruct.out_path,'kinematic_feats.h5']))

# pc_wlet, pc_wlet_labels = pca(wlet_feats,
#                               wlet_labels,
#                               categories = ['wlet_vel','wlet_ego_euc','wlet_ang','wlet_avel'],
#                               n_pcs = 10,
#                               method = 'fbpca')

# del wlet_feats, wlet_labels

# pc_feats = np.hstack((pc_feats, pc_wlet))
# pc_labels += pc_wlet_labels
# del pc_wlet, pc_wlet_labels

# save_h5(pc_feats, pc_labels, path = ''.join([pstruct.out_path,'pca_feats.h5']))

# # pc_feats, pc_labels = read_h5(path = ''.join([pstruct.out_path,'pca_feats.h5']))

# params = itf.read_params_config(config_path = '../configs/fitsne.yaml')

# pstruct.features = pc_feats[::params['downsample']]
# pstruct.frame_id = np.arange(0,pc_feats.shape[0],params['downsample'])
# pstruct.exp_id = pstruct.exp_ids_full[::params['downsample']]
# pstruct.downsample = params['downsample']
# pstruct.load_meta()

# itf.run_analysis(params_config = '../configs/fitsne.yaml',
#                  ds = pstruct)

# print("All Done! Total Time: ")
# print(time.time() - t)


import pickle
pstruct = pickle.load(open('/home/exx/Desktop/GitHub/results/R01_all/fitsne/datastruct.p','rb'))

features, labels =  read_h5(path = '/home/exx/Desktop/GitHub/results/R01_all/postural_feats.h5')
features = features[::5, :]

# Standard Scaling
features -= features.mean(axis=0)
feat_std = np.std(features,axis=0)
features = features[:,feat_std!=0]
features = features/feat_std[feat_std!=0]
labels = [label for i, label in enumerate(labels) if feat_std[i]!=0]

vis.density_feat(pstruct, pstruct.ws, features, labels, 'avel_0_3_4_xy_5')