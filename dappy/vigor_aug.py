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
from scipy.spatial import distance

analysis_key = "vigor"
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

    ## Vigor augmentation
    vigor_level = np.linspace(0.5, 2, 10)
    pose_augmented = np.empty((0, pose.shape[1], pose.shape[2]))
    new_id = np.empty(0)
    new_id_counter = 0
    for _, i in enumerate(tqdm(np.unique(id))):
        tot_frames = np.sum(id == i)
        cs = CubicSpline(np.arange(tot_frames), pose[id == i, ...])
        ex_frame = 1000

        # if i==id[0]:
        #     vis.skeleton_vid3D(
        #         pose[id==i,...],
        #         connectivity,
        #         frames=[ex_frame],
        #         N_FRAMES=200,
        #         dpi=100,
        #         VID_NAME=f"example_vid{i}_frame{ex_frame}.mp4",
        #         SAVE_ROOT="".join([paths["out_path"], params["label"]]),
        #     )

        for level in vigor_level:
            new_t = np.linspace(0, tot_frames - 1, int(tot_frames / level))
            pose_interpolated = cs(new_t)

            pose_augmented = np.append(pose_augmented, pose_interpolated, axis=0)

            new_id = np.append(
                new_id, np.ones(pose_interpolated.shape[0]) * new_id_counter
            )
            new_id_counter += 1

            # if i==id[0]:
            #     vis.skeleton_vid3D(
            #         pose_interpolated,
            #         connectivity,
            #         frames=[int(ex_frame/level)],
            #         N_FRAMES=200,
            #         dpi=100,
            #         VID_NAME=f"vigor{level:.2}_vid{i}_frame{ex_frame}.mp4",
            #         SAVE_ROOT="".join([paths["out_path"], params["label"]]),
            #     )

    meta_expanded = meta.iloc[
        np.repeat(np.arange(np.max(id) + 1), len(vigor_level))
    ].reset_index(drop=True)
    meta_expanded["vigor"] = np.tile(vigor_level, np.max(id) + 1)
    meta_expanded["old_id"] = meta_expanded["id"].copy()
    meta_expanded["id"] = meta_expanded.index
    meta_by_frame_expanded = meta_expanded.iloc[new_id].reset_index(drop=True)

    pose = rotate_spine(center_spine(pose_augmented))
    features, labels = run.standard_features(pose, connectivity, paths)
    data_obj = run.run(
        features,
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

vigor_level = np.linspace(0.5, 2, 10)
dist_js = analysis.bin_embed_distance(
    values=data_obj.embed_vals,
    meta=np.array(data_obj.data["vigor"]),
    augmentation=vigor_level,
    time_bins=100,
    hist_bins=100,
    hist_range=data_obj.ws.hist_range,
)

f = plt.figure()
plt.plot(vigor_level[1:], dist_js,'k.')
plt.xlabel("X Speed")
plt.ylabel("Jensen-Shannon Distance")
plt.savefig("".join([paths["out_path"], params["label"], "/aug_distances.png"]))


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
#     cat2="vigor",
#     watershed=data_obj.ws,
#     filepath="".join([paths["out_path"], params["label"], "/density_grid.png"]),
# )

# vis.skeleton_vid3D_cat(
#     data_obj,
#     "Cluster",
#     n_skeletons=10,
#     filepath="".join([paths["out_path"], params["label"], "/"]),
# )

# import pdb; pdb.set_trace()
