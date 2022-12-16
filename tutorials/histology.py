from ..src.features import *
import ..src.DataStruct as ds
import ..src.visualization as vis
import ..src.interface as itf
import numpy as np
import time
import ..src.read, ..src.write
from ..src.embed import Watershed, Embed
import pickle
import ..src.analysis

analysis_key = "histology_no24"
paths = read.config("../configs/path_configs/" + analysis_key + ".yaml")
params = read.config("../configs/param_configs/fitsne.yaml")
connectivity = read.connectivity(
    path=paths["skeleton_path"], skeleton_name=paths["skeleton_name"]
)

# pose, id, meta, meta_by_frame = read.pose_from_meta(
#     path=paths["meta_path"], connectivity=connectivity
# )

# # pose, vid_id = read.pose(paths["pose_path"], connectivity, paths["exp_key"])
# pose, vid_id = read.pose_h5(
#     paths["data_path"] + "pose_fixed_aligned.h5", paths["exp_key"]
# )

# # # Separate videos have rotated floor planes - this rotates them back
# # pose = align_floor(pose, vid_id)

# # # Cleaning up some bad tracking frames
# # pose = median_filter(pose, vid_id, filter_len=5)  # Regular median filter
# # pose = z_filter(pose, vid_id, threshold=2000)  # , connectivity = connectivity)
# # pose = vel_filter(
# #     pose, vid_id, threshold=100
# # )  # ,max_iter=10, connectivity = connectivity) # Finds location of high velocity, removes, and interpolates value

# # write.pose_h5(
# #     pose, vid_id, paths["exp_key"], paths["data_path"] + "pose_fixed_aligned.h5"
# # )

# # # Calculating absolute velocities and standard deviation of velocites over windows
# # abs_vel, abs_vel_labels = get_velocities_fast(
# #     pose, vid_id, connectivity.joint_names, joints=[0, 4, 5]
# # )

# # Centering all joint locations to mid-spine then Rotates front spine to xz axis
# pose = rotate_spine(center_spine(pose))

# # # Getting relative velocities
# # rel_vel, rel_vel_labels = get_velocities_fast(
# #     pose, vid_id, connectivity.joint_names, joints=np.delete(np.arange(18), 4)
# # )
# # features = np.concatenate([abs_vel, rel_vel],axis=1)
# # labels = abs_vel_labels + rel_vel_labels
# # write.features_h5(
# #     features,
# #     labels,
# #     path="".join([paths["data_path"], "vel_feats.h5"]),
# # )

# # del abs_vel, rel_vel, abs_vel_labels, rel_vel_labels

# # Reshape pose to get egocentric pose features
# ego_pose, ego_pose_labels = get_ego_pose(pose, connectivity.joint_names)
# ego_pose, ego_pose_labels = standard_scale(ego_pose, ego_pose_labels)

# # Calculating joint angles
# angles, angle_labels = get_angles(pose, connectivity.angles)

# # # Calculating angle velocities
# # ang_vel, ang_vel_labels = get_angular_vel_fast(angles, angle_labels, vid_id)

# # Collect all features together
# labels = ego_pose_labels + angle_labels
# features = np.concatenate([ego_pose, angles], axis=1)

# # Save postural features to h5 file
# write.features_h5(features, labels, path="".join([paths["data_path"], "postural_feats_scaled.h5"]))

# # Read postural features from h5 file
# features, labels = read.features_h5(
#     path="".join([paths["data_path"], "postural_feats_scaled.h5"])
# )

# feat_categories = ["ego_euc", "ang"]
# # # feat_categories = ["abs_vel", "rel_vel", "ego_euc", "ang", "avel"]

# pc_feats, pc_labels = pca(
#     features, labels, categories=feat_categories, n_pcs=8, method="fbpca"
# )

# del features, labels
# write.features_h5(pc_feats, pc_labels, path="".join([paths["out_path"], "pc_postural_feats_scaled.h5"]))
# del pc_feats, pc_labels
# # import pdb; pdb.set_trace()

# # wlet_feats, wlet_labels = wavelet(
# #     pc_feats,
# #     pc_labels,
# #     vid_id,
# #     sample_freq=90,
# #     freq=np.linspace(0.5, 25, 25),
# #     #   freq = np.linspace(0.5,4.5,25)**2,
# #     w0=5,
# # )

# # write.features_h5(
# #     wlet_feats, wlet_labels, path="".join([paths["out_path"], "kinematic_feats_scaled.h5"])
# # )

# wlet_feats, wlet_labels = read.features_h5(path="".join([paths["out_path"], "kinematic_feats_scaled.h5"]))
# # import pdb; pdb.set_trace()
# pc_wlet, pc_wlet_labels = pca(
#     wlet_feats,
#     wlet_labels,
#     categories=["wlet_" + cat for cat in feat_categories],
#     n_pcs=8,
#     method="fbpca",
# )

# del wlet_feats, wlet_labels

# pc_feats, pc_labels = read.features_h5("".join([paths["out_path"], "pc_postural_feats_scaled.h5"]))

# pc_feats = np.hstack((pc_feats, pc_wlet))
# pc_labels += pc_wlet_labels
# del pc_wlet, pc_wlet_labels

# write.features_h5(
#     pc_feats, pc_labels, path="".join([paths["out_path"], "pca_feats_scaled.h5"])
# )

pose, vid_id = read.pose_h5(
    paths["data_path"] + "pose_fixed_aligned.h5", paths["exp_key"]
)

meta, meta_by_frame = read.meta(paths["meta_path"], id=vid_id)

data_obj = ds.DataStruct(
    pose=pose,
    id=vid_id,
    id_full=vid_id,
    meta=meta,
    meta_by_frame=meta_by_frame,
    connectivity=connectivity,
)

pc_feats, pc_labels = read.features_h5(path = ''.join([paths['out_path'],'pca_feats.h5']))
data_obj.features = pc_feats
data_obj = data_obj[:: params["downsample"], :]

# Embedding using fitsne
embedder = Embed(
    embed_method=params["single_embed"]["method"],
    perplexity=params["single_embed"]["perplexity"],
    lr=params["single_embed"]["lr"],
)
data_obj.embed_vals = embedder.embed(data_obj.features, save_self=True)

# Watershed clustering
data_obj.ws = Watershed(
    sigma=params["single_embed"]["sigma"], max_clip=1, log_out=True, pad_factor=0.05
)
data_obj.data.loc[:, "Cluster"] = data_obj.ws.fit_predict(data=data_obj.embed_vals)

# print("Writing Data Object to pickle")
data_obj.write_pickle("".join([paths["out_path"], params["label"], "/"]))
# data_obj = pickle.load(
#     open("".join([paths["out_path"], params["label"], "/datastruct.p"]), "rb")
# )

vis.density(
    data_obj.ws.density,
    data_obj.ws.borders,
    filepath="".join([paths["out_path"], params["label"], "/density.png"]),
    show=False,
)
vis.scatter(
    data_obj.embed_vals,
    filepath="".join([paths["out_path"], params["label"], "/scatter.png"]),
)

for cat in params["density_by_column"]:
    vis.density_cat(
        data=data_obj,
        column=cat,
        watershed=data_obj.ws,
        n_col=4,
        filepath="".join(
            [paths["out_path"], params["label"], "/density_", cat, ".png"]
        ),
    )

vis.density_grid(
    data=data_obj,
    cat1="Condition",
    cat2="AnimalID",
    watershed=data_obj.ws,
    filepath="".join([paths["out_path"], params["label"], "/density_grid.png"]),
)

vis.skeleton_vid3D_cat(
    data_obj,
    "Cluster",
    n_skeletons=10,
    filepath="".join([paths["out_path"], params["label"], "/"]),
)

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

# vis.labeled_watershed(
#     data_obj.ws.watershed_map,
#     paths["out_path"] + params["label"] + "/behavior_labels.csv",
# )

# vis.density_feat(data_obj, data_obj.ws, features, labels, "avel_0_3_4_xy_5")
