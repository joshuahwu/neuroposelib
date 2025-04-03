import pickle
from neuroposelib import vis, read, preprocess
import numpy as np

analysis_key = "ensemble_healthy"
config = read.config("../configs/" + analysis_key + ".yaml")

pose, ids = read.pose_h5(config["data_path"] + "pose_aligned.h5", dtype=np.float32)

connectivity = read.connectivity(
    path=config["skeleton_path"], skeleton_name=config["skeleton_name"]
)
data_obj = pickle.load(open("".join([config["out_path"], "/datastruct.p"]), "rb"))

vis.pose.sample_grid3D(
    pose-pose.mean(axis=-2, keepdims=True),
    connectivity=connectivity,
    labels=data_obj.data["Cluster"],
    n_samples=9,
    centered=True,
    N_FRAMES=100,
    fps=90,
    dpi=100,
    watershed=data_obj.ws,
    embed_vals=None,
    VID_NAME = "cluster",
    filepath=config["out_path"],
)

vis.plt.density(
    data_obj.ws.density,
    data_obj.ws.borders,
    filepath="".join([config["out_path"], "/density.png"]),
    show=False,
)

vis.plot.scatter(
    data_obj.embed_vals,
    filepath="".join([config["out_path"], "/scatter.png"]),
)

for cat in ["id", "Sex", "Cluster"]:
    vis.plt.density_cat(
        data=data_obj,
        column=cat,
        watershed=data_obj.ws,
        filepath="".join([config["out_path"], "/density_", cat, ".png"]),
    )

vis.plt.density_grid(
    data=data_obj,
    cat1="Condition",
    cat2="id",
    watershed=data_obj.ws,
    filepath="".join([config["out_path"], "/density_grid.png"]),
)

import pdb

pdb.set_trace()
