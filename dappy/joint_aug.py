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
import run
from pathlib import Path
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler

analysis_key = "joint"
paths = read.config("../configs/path_configs/" + analysis_key + ".yaml")
params = read.config("../configs/param_configs/fitsne.yaml")

if Path(paths["out_path"] + params["label"] + "/datastruct.p").exists():
    data_obj = pickle.load(
        open("".join([paths["out_path"], params["label"], "/datastruct.p"]), "rb")
    )

else:
    connectivity = read.connectivity(
        path=paths["skeleton_path"], skeleton_name=paths["skeleton_name"]
    )

    pose, id = read.pose_h5(paths["data_path"] + "pose_aligned.h5")
    pose = pose[::2, ...]
    id = id[::2, ...]

    meta, meta_by_frame = read.meta(paths["meta_path"], id=id)

    pose = rotate_spine(center_spine(pose))
    features, labels = run.standard_features(pose, connectivity, paths)

    # features = StandardScaler.fit_transform(features)
    feat_df = pd.DataFrame(features, columns=labels)

    ## Body plan augmentation
    joint_groups = {
        "Ears": [1, 2],
        "Paws": [6, 9, 12, 15],
        "Wrists": [7, 10, 13, 16],
        "Mid-legs": [8, 11, 14, 17],
    }

    keys = list(joint_groups.keys())
    num_frames = features.shape[0]

    meta_expanded = meta.copy()
    meta_expanded["joint_ablation"] = "Normal"
    new_id = id

    cum_joint_bool = np.zeros(len(feat_df.columns)).astype(bool)
    ## Single group ablation
    for i, key in enumerate(joint_groups.keys()):
        joint_names = [connectivity.joint_names[i] for i in joint_groups[key]]
        val_1uscore = ["_" + str(i) for i in joint_groups[key]]
        key_search = joint_names + (["_" + str(i) + "_" for i in joint_groups[key]])

        temp_feat = feat_df.iloc[:num_frames].copy()

        ## Isolate features which utilize
        contains_joint = feat_df.columns.str.contains(
            "|".join(key_search)
        ) | feat_df.columns.str.endswith(tuple(val_1uscore))

        ## Set those columns to the mean
        means = np.array(temp_feat.iloc[:, contains_joint].mean(axis=0))[None, ...]
        means = np.repeat(means, num_frames, axis=0)
        temp_feat.iloc[:, contains_joint] = means

        feat_df = feat_df.append(temp_feat)

        ## Adjust metadata
        new_id = np.append(new_id, id + (i + 1) * len(np.unique(id)))
        meta_temp = meta.copy()
        meta_temp["joint_ablation"] = key
        meta_expanded = meta_expanded.append(meta_temp).reset_index(drop=True)

        ## Cumulative joint ablation
        if i > 0:
            cum_joint_bool = cum_joint_bool | contains_joint

            temp_feat = feat_df.iloc[:num_frames].copy()
            means = np.array(temp_feat.iloc[:, cum_joint_bool].mean(axis=0))[None, ...]
            means = np.repeat(means, num_frames, axis=0)

            temp_feat.iloc[:, cum_joint_bool] = means

            meta_temp = meta_temp = meta.copy()
            meta_temp["joint_ablation"] = "_".join(list(joint_groups.keys())[0 : i + 1])

            if i == 1:
                cum_ablate_feat = temp_feat.copy()
                cum_id = id + (i + len(joint_groups.keys())) * len(np.unique(id))
                cum_meta = meta_temp.copy()
            else:
                cum_ablate_feat = cum_ablate_feat.append(temp_feat)
                cum_id = np.append(
                    cum_id, id + (i + len(joint_groups.keys())) * len(np.unique(id))
                )
                cum_meta = cum_meta.append(meta_temp)
        # import pdb; pdb.set_trace()

    ## Concatenate single ablations and cumulative ablations together
    new_id = np.append(new_id, cum_id)
    meta_expanded = meta_expanded.append(cum_meta).reset_index(drop=True)
    feat_df = feat_df.append(cum_ablate_feat).reset_index(drop=True)
    meta_expanded["old_id"] = meta_expanded["id"].copy()
    meta_expanded["id"] = meta_expanded.index
    meta_by_frame_expanded = meta_expanded.iloc[new_id].reset_index(drop=True)

    num_augments = len(np.unique(meta_expanded["joint_ablation"]))
    pose_augmented = np.moveaxis(np.tile(pose[..., None], num_augments), -1, 0).reshape(
        (pose.shape[0] * num_augments, pose.shape[1], pose.shape[2])
    )
    data_obj = run.run(
        feat_df.to_numpy(),
        labels,
        pose_augmented,
        new_id,
        connectivity,
        paths,
        params,
        meta_expanded,
        meta_by_frame_expanded,
    )
    # print("Writing Data Object to pickle")
    data_obj.write_pickle("".join([paths["out_path"], params["label"], "/"]))

aug_levels = list(data_obj.meta["joint_ablation"].unique())
dist_js = analysis.bin_embed_distance(
    values=data_obj.embed_vals,
    meta=np.array(data_obj.data["joint_ablation"]),
    augmentation=aug_levels,
    time_bins=100,
    hist_bins=100,
    hist_range=data_obj.ws.hist_range,
)


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
#     cat1="old_id",
#     cat2="joint_ablation",
#     watershed=data_obj.ws,
#     filepath="".join([paths["out_path"], params["label"], "/density_grid.png"]),
# )

# vis.skeleton_vid3D_cat(
#     data_obj,
#     "Cluster",
#     n_skeletons=10,
#     filepath="".join([paths["out_path"], params["label"], "/"]),
# )

dist_df = pd.DataFrame(dist_js,columns=["Jensen_Shannon"],index=aug_levels[1:])

import seaborn as sns
sns.set(rc={'figure.figsize':(12,8)})
js_plot = sns.lineplot(dist_df.reset_index(),x='index',y='Jensen_Shannon')
plt.xticks(rotation=45)
plt.tight_layout()
js_plot.get_figure().savefig("".join([paths["out_path"], params["label"], "/aug_distances.png"]))
# plt.xlabel("Joint Ablations")
# plt.ylabel("Jensen-Shannon Distance")
# plt.savefig("".join([paths["out_path"], params["label"], "/aug_distances.png"]))

# # import pdb; pdb.set_trace()
