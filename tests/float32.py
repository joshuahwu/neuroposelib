from neuroposelib import read, write, features, analysis, preprocess
import neuroposelib.visualization as vis
from neuroposelib.embed import Watershed, Embed
import neuroposelib.DataStruct as ds
import numpy as np

analysis_key = "ensemble_healthy"
config = read.config("../configs/" + analysis_key + ".yaml")

connectivity = read.connectivity(
    path=config["skeleton_path"], skeleton_name=config["skeleton_name"]
)

# pose, ids, meta, meta_by_frame = read.pose_from_meta(
#     config["meta_path"],
#     connectivity=connectivity,
#     dtype=np.float32,
#     key="ClusterDirectory",
# )

pose, ids = read.pose_h5(config["data_path"] + "pose_aligned.h5", dtype=np.float32)

meta, meta_by_frame = read.meta(config["meta_path"], id=ids)
assert pose.dtype == np.float32

# Separate videos have rotated floor planes - this rotates them back
pose = preprocess.align_floor_by_id(pose, ids)
assert pose.dtype == np.float32

pose = preprocess.median_filter(pose, ids, filter_len=5)  # Regular median filter
assert pose.dtype == np.float32

# vis.pose3D_arena(
#     pose=pose,
#     connectivity=connectivity,
#     frames=[3e4, 4e5, 8e5, 1e6],
#     centered=False,
#     N_FRAMES=100,
#     fps=90,
#     dpi=100,
#     VID_NAME="align_arena.mp4",
#     SAVE_ROOT="./test_plots/",
# )

# vis.pose3D_grid(
#     pose=pose,
#     connectivity=connectivity,
#     frames=[3e4, 4e5, 8e5, 1e6],
#     centered=False,
#     N_FRAMES=100,
#     fps=90,
#     dpi=100,
#     VID_NAME="align_grid.mp4",
#     SAVE_ROOT="./test_plots/",
# )

# Calculating velocities and standard deviation of velocites over windows
abs_vel, abs_vel_labels = features.get_velocities(
    pose,
    ids,
    connectivity.joint_names,
    joints=[0, 4, 5],
    widths=[3, 31, 89],
    abs_val=False,
    f_s=90,
    std=True,
)
assert abs_vel.dtype == np.float32

# Centering all joint locations to mid-spine
pose = preprocess.center_spine(pose)
assert pose.dtype == np.float32

# Rotates front spine to xz axis
pose = preprocess.rotate_spine(pose)
assert pose.dtype == np.float32

# Reshape pose to get egocentric pose features
ego_pose, ego_pose_labels = features.get_ego_pose(pose, connectivity.joint_names)
assert ego_pose.dtype == np.float32

# Calculating joint angles
angles, angle_labels = features.get_angles(pose, connectivity.angles)
assert angles.dtype == np.float32

# Calculating angle velocities
ang_vel, ang_vel_labels = features.get_angular_vel(angles, angle_labels, ids)
assert ang_vel.dtype == np.float32

# Collect all features together
labels = ego_pose_labels + angle_labels
feats = np.concatenate([ego_pose, angles], axis=1)
assert feats.dtype == np.float32

feat_categories = ["ego_euc", "ang"]

pc_feats, pc_labels = features.pca(
    feats, labels, categories=feat_categories, n_pcs=5, method="fbpca"
)
assert pc_feats.dtype == np.float32

wlet_feats, wlet_labels = features.wavelet(
    features=pc_feats,
    labels=pc_labels,
    ids=ids,
    f_s=90,
    freq=np.linspace(0.5, 4.5, 25) ** 2,
    w0=5,
)
assert wlet_feats.dtype == np.float32

pc_wlet, pc_wlet_labels = features.pca(
    wlet_feats,
    wlet_labels,
    categories=["wlet_" + cat for cat in feat_categories],
    n_pcs=5,
    method="fbpca",
)
assert pc_wlet.dtype == np.float32

pc_feats = np.hstack((pc_feats, pc_wlet))
pc_labels += pc_wlet_labels

data_obj = ds.DataStruct(
    pose=pose, id=ids, meta=meta, meta_by_frame=meta_by_frame, connectivity=connectivity
)
data_obj.features = pc_feats
data_obj = data_obj[:: config["downsample"], :]

# Embedding using t-SNE
embedder = Embed(
    embed_method=config["single_embed"]["method"],
    perplexity=config["single_embed"]["perplexity"],
    lr=config["single_embed"]["lr"],
)
data_obj.embed_vals = embedder.embed(data_obj.features, save_self=True)
assert data_obj.embed_vals.dtype == np.float32

# Watershed clustering
data_obj.ws = Watershed(
    sigma=config["single_embed"]["sigma"], max_clip=1, log_out=True, pad_factor=0.05
)

data_obj.data.loc[:, "Cluster"] = data_obj.ws.fit_predict(data=data_obj.embed_vals)
data_obj.write_pickle(config["out_path"])
