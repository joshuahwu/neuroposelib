import pickle
from dappy import vis, read

analysis_key = "ensemble_healthy"
config = read.config("../configs/" + analysis_key + ".yaml")

connectivity = read.connectivity(
    path=config["skeleton_path"], skeleton_name=config["skeleton_name"]
)
data_obj = pickle.load(
        open("".join([config["out_path"], "/datastruct.p"]), "rb")
    )

import pdb; pdb.set_trace()