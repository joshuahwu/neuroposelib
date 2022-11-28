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

analysis_key = "histology_no24"
paths = read.config("../configs/path_configs/" + analysis_key + ".yaml")
params = read.config("../configs/param_configs/fitsne.yaml")
connectivity = read.connectivity(
    path=paths["skeleton_path"], skeleton_name=paths["skeleton_name"]
)

# _, vid_id = read.pose_h5(paths["data_path"] + "pose_fixed_aligned.h5", paths["exp_key"])

# # Calculating absolute velocities and standard deviation of velocites over windows
# abs_vel, abs_vel_labels = get_velocities_fast(
#     pose, vid_id, connectivity.joint_names, joints=[0, 4, 5], std=False
# )
# norm_ind = [i for i, label in enumerate(abs_vel_labels) if "norm" in label]
# mean_abs_vel = np.mean(abs_vel[:, norm_ind], axis=1)
# vis.feature_hist(mean_abs_vel, label='abs_vel', filepath = "".join([paths["out_path"], params["label"], "/abs_vel_hist.png"]))

# # Centering all joint locations to mid-spine then Rotates front spine to xz axis
# pose = rotate_spine(center_spine(pose))
# # Getting relative velocities
# rel_vel, rel_vel_labels = get_velocities_fast(
#     pose, vid_id, connectivity.joint_names, joints=np.delete(np.arange(18), 4), std=False
# )

# norm_ind = [i for i, label in enumerate(abs_vel_labels) if "norm" in label]
# mean_rel_vel = np.mean(rel_vel[:,norm_ind], axis=1)
# vis.feature_hist(mean_rel_vel, label='rel_vel', filepath = "".join([paths["out_path"], params["label"], "/rel_vel_hist.png"]))

# # Read postural features from h5 file
# vel_feats, vel_labels = read.features_h5(
#     path="".join([paths["data_path"], "vel_feats.h5"])
# )

# abs_vel_label_ind = [
#     i
#     for i, label in enumerate(vel_labels)
#     if (("abs_vel" in label) and ("norm" in label))
# ]
# rel_vel_label_ind = [
#     i
#     for i, label in enumerate(vel_labels)
#     if (("rel_vel" in label) and ("norm" in label))
# ]
# abs_vel_labels = [vel_labels[i] for i in abs_vel_label_ind]
# rel_vel_labels = [vel_labels[i] for i in rel_vel_label_ind]
# abs_vel_feats = vel_feats[:, abs_vel_label_ind]
# rel_vel_feats = vel_feats[:, rel_vel_label_ind]
# del vel_feats, vel_labels

# mean_abs_vel = np.mean(standard_scale(abs_vel_feats, abs_vel_labels)[0], axis=1)
# mean_rel_vel = np.mean(standard_scale(rel_vel_feats, rel_vel_labels)[0], axis=1)

# write.features_h5(
#     np.append(mean_abs_vel[:, None], mean_rel_vel[:, None], axis=1),
#     labels=["mean_abs_vel", "mean_rel_vel"],
#     path="".join([paths["data_path"], "mean_vels.h5"]),
# )

# # Read postural features from h5 file
# features, labels = read.features_h5(
#     path="".join([paths["data_path"], "postural_feats_scaled.h5"])
# )

# feat_categories = ["ego_euc", "ang"]

# pc_feats, pc_labels = pca(
#     features, labels, categories=feat_categories, n_pcs=8, method="fbpca"
# )

# del features, labels
# write.features_h5(
#     pc_feats, pc_labels, path="".join([paths["out_path"], "pc_postural_active.h5"])
# )
# del pc_feats, pc_labels
# # import pdb; pdb.set_trace()

# wlet_feats, wlet_labels = wavelet(
#     pc_feats,
#     pc_labels,
#     vid_id,
#     sample_freq=90,
#     freq=np.linspace(0.5, 25, 25),
#     #   freq = np.linspace(0.5,4.5,25)**2,
#     w0=5,
# )

# # write.features_h5(
# #     wlet_feats, wlet_labels, path="".join([paths["out_path"], "kinematic_feats_scaled.h5"])
# # )

# wlet_feats, wlet_labels = read.features_h5(
#     path="".join([paths["out_path"], "kinematic_feats_active.h5"])
# )
# # import pdb; pdb.set_trace()
# pc_wlet, pc_wlet_labels = pca(
#     wlet_feats,
#     wlet_labels,
#     categories=["wlet_" + cat for cat in feat_categories],
#     n_pcs=8,
#     method="fbpca",
# )

# del wlet_feats, wlet_labels

# pc_feats, pc_labels = read.features_h5(
#     "".join([paths["out_path"], "pc_postural_active.h5"])
# )

# pc_feats = np.hstack((pc_feats, pc_wlet))
# pc_labels += pc_wlet_labels
# del pc_wlet, pc_wlet_labels

# write.features_h5(
#     pc_feats, pc_labels, path="".join([paths["out_path"], "pca_active.h5"])
# )

# pose, vid_id = read.pose_h5(
#     paths["data_path"] + "pose_fixed_aligned.h5", paths["exp_key"]
# )

# meta, meta_by_frame = read.meta(paths["meta_path"], id=vid_id)

# data_obj = ds.DataStruct(
#     pose=pose,
#     id=vid_id,
#     id_full=vid_id,
#     meta=meta,
#     meta_by_frame=meta_by_frame,
#     connectivity=connectivity,
# )

# pc_feats, pc_labels = read.features_h5(
#     path="".join([paths["out_path"], "pca_feats_scaled.h5"])
# )
# data_obj.features = pc_feats

# ## Cleaning out frames of data based on velocity and also removing edge frames
# non_edge_ind = remove_edge_ids(vid_id, 100)
# import pdb; pdb.set_trace()

# vel_scores, _ = read.features_h5(
#     path="".join([paths["data_path"], "mean_vels.h5"])
# )
# vel_scores = vel_scores[non_edge_ind,:]

# active_ind = np.where(np.logical_and(-0.5 < vel_scores[:, 0], vel_scores[:, 0] < 3) & np.logical_and(-0.5 < vel_scores[:, 1],vel_scores[:,1]<3))[0]

# vis.feature_hist(vel_scores[:,0], "Abs_Vel_Score", filepath = paths["out_path"], range=(vel_scores[:,0].min(), 3))
# vis.feature_hist(vel_scores[:,1], "Rel_Vel_Score", filepath = paths["out_path"], range=(vel_scores[:,1].min(), 3))

# data_obj = data_obj[non_edge_ind, :]
# data_obj = data_obj[active_ind, :]
# data_obj = data_obj[:: params["downsample"], :]

# # Embedding using fitsne
# embedder = Embed(
#     embed_method=params["single_embed"]["method"],
#     perplexity=params["single_embed"]["perplexity"],
#     lr=params["single_embed"]["lr"],
# )
# data_obj.embed_vals = embedder.embed(data_obj.features, save_self=True)

data_obj = pickle.load(
    open("".join([paths["out_path"], params["label"], "/datastruct.p"]), "rb")
)
# # Watershed clustering
# data_obj.ws = Watershed(
#     sigma=params["single_embed"]["sigma"], max_clip=1, log_out=True, pad_factor=0.05
# )
# data_obj.data.loc[:, "Cluster"] = data_obj.ws.fit_predict(data=data_obj.embed_vals)

# print("Writing Data Object to pickle")
# data_obj.write_pickle("".join([paths["out_path"], params["label"], "/"]))

# vis.density(
#     data_obj.ws.density,
#     data_obj.ws.borders,
#     filepath="".join([paths["out_path"], params["label"], "/density.png"]),
#     show=False,
# )
# vis.scatter(
#     data_obj.embed_vals,
#     filepath="".join([paths["out_path"], params["label"], "/scatter.png"]),
# )

# for cat in params["density_by_column"]:
#     vis.density_cat(
#         data=data_obj,
#         column=cat,
#         watershed=data_obj.ws,
#         n_col=4,
#         filepath="".join(
#             [paths["out_path"], params["label"], "/density_", cat, ".png"]
#         ),
#     )

# vis.density_grid(
#     data=data_obj,
#     cat1="Condition",
#     cat2="AnimalID",
#     watershed=data_obj.ws,
#     filepath="".join([paths["out_path"], params["label"], "/density_grid.png"]),
# )

# vis.skeleton_vid3D_cat(
#     data_obj,
#     "Cluster",
#     n_skeletons=10,
#     filepath="".join([paths["out_path"], params["label"], "/"]),
# )

# features, labels = read.features_h5(path=paths["data_path"] + "/postural_feats.h5")
# features = features[:: params["downsample"], :]
# features, labels = standard_scale(features, labels)

# heur = read.heuristics(path=paths["heuristics_path"])

# vis.heuristics(
#     features,
#     labels,
#     data_obj,
#     heur.HEURISTICS_DICT[paths["skeleton_name"]],
#     filepath=paths["out_path"] + params["label"],
# )

vis.labeled_watershed(
    data_obj.ws.watershed_map,
    data_obj.ws.borders,
    paths["out_path"] + params["label"] + "/behavior_labels.csv",
)

# vis.density_feat(data_obj, data_obj.ws, features, labels, "avel_0_3_4_xy_5")
