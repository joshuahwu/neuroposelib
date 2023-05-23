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

analysis_key = "noise"
paths = read.config("../configs/path_configs/" + analysis_key + ".yaml")
params = read.config("../configs/param_configs/fitsne.yaml")

if Path(paths['out_path'] + params["label"]+'/datastruct.p').exists():
    data_obj = pickle.load(
        open("".join([paths["out_path"], params["label"], "/datastruct.p"]), "rb")
    )
    
else:
    connectivity = read.connectivity(
        path=paths["skeleton_path"], skeleton_name=paths["skeleton_name"]
    )

    pose, id = read.pose_h5(paths['data_path']+'pose_aligned.h5')
    pose = pose[::2,...]
    id = id[::2,...]

    meta, meta_by_frame = read.meta(paths['meta_path'],
                                    id = id)

    ## Noise augmentation
    noise_level = np.linspace(0, 3, 10)
    pose_augmented = np.repeat(pose,len(noise_level),axis=0)
    noise = np.zeros(np.shape(pose),dtype=np.float32)
    for level in noise_level[1:]:
        rng = np.random.default_rng()
        noise = np.append(noise,np.random.normal(0,level,np.shape(pose)).astype(np.float32),axis=0)

    pose_augmented += noise

    vid_len = len(meta_by_frame.index[meta_by_frame['id']==0])

    new_id = np.add.outer(id, np.arange(0,len(noise_level),dtype=np.int16)*(np.max(id)+1)).T.reshape((-1,))

    meta_expanded = meta.iloc[np.tile(np.arange(np.max(id)+1), len(noise_level))].reset_index(drop=True)
    meta_expanded['noise'] = np.repeat(noise_level,np.max(id)+1)
    meta_expanded['old_id'] = meta_expanded['id'].copy()
    meta_expanded['id'] = meta_expanded.index
    meta_by_frame_expanded = meta_expanded.iloc[new_id].reset_index(drop=True)

    pose = rotate_spine(center_spine(pose_augmented))

    features, labels = run.standard_features(pose, connectivity, paths)

    data_obj = run.run(features, labels, pose_augmented, new_id, connectivity,paths,params,meta_expanded,meta_by_frame_expanded)
    # print("Writing Data Object to pickle")
    data_obj.write_pickle("".join([paths["out_path"], params["label"], "/"]))


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
#     cat2="noise",
#     watershed=data_obj.ws,
#     filepath="".join([paths["out_path"], params["label"], "/density_grid.png"]),
# )


noise_level = np.linspace(0, 3, 10)
dist_js = analysis.bin_embed_distance(
    values=data_obj.embed_vals,
    meta=np.array(data_obj.data["noise"]),
    augmentation=noise_level,
    time_bins=100,
    hist_bins=100,
    hist_range=data_obj.ws.hist_range,
)

f = plt.figure()
plt.plot(noise_level[1:], dist_js,'k.')
plt.xlabel("Gaussian Noise (mm)")
plt.ylabel("Jensen-Shannon Distance")
plt.savefig("".join([paths["out_path"], params["label"], "/aug_distances.png"]))

# vis.skeleton_vid3D_cat(
#     data_obj,
#     "Cluster",
#     n_skeletons=10,
#     filepath="".join([paths["out_path"], params["label"], "/"]),
# )