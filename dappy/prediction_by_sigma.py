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
from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import StandardScaler
import seaborn as sns

analysis_key = "histology_no24"
paths = read.config("../configs/path_configs/" + analysis_key + ".yaml")
params = read.config("../configs/param_configs/fitsne.yaml")

SCORES = np.array(
    [
        26.310916895326,
        14.4640686845522,
        14.6128769719533,
        14.25316120785,
        25.0672411290027,
        11.9516110337281,
        14.4139133290107,
        12.7854028107235,
        13.0827375133176,
        24.9018692592874,
        12.1369302259479,
        15.3167937667823,
        9.26316678695589,
        9.56779770449291,
        10.1409421926758,
        12.3750133149208,
        8.88350998195046,
        7.36888390813476,
        10.6822128015899,
        8.42901405057676,
        9.11480854747771,
        10.0537277517741,
        9.05169239186871,
        10.6387127484618,
        11.888031361915,
        10.8120940463736,
        10.3971083521533,
        7.21074466554142,
        6.66613202563015,
        6.40222987251535,
        17.3597973028407,
        11.0008367316835,
        10.6683739069522,
        13.0084053141184,
        12.929530206104,
        14.5476263045027,
        13.4522986526429,
    ]
)

data_obj = pickle.load(
    open("".join([paths["out_path"], params["label"], "/datastruct.p"]), "rb")
)

sigma_list = np.linspace(5, 25, 20)
r2 = np.zeros(np.shape(sigma_list))
for i, sigma in enumerate(sigma_list):
    ws = Watershed(sigma=sigma, max_clip=1, log_out=True, pad_factor=0.05)
    cluster_ids_by_frame = ws.fit_predict(data=data_obj.embed_vals)
    if i == len(sigma_list)-1:
        vis.density(
        ws.density,
        ws.borders,
        filepath="".join([paths["out_path"], params["label"], "/density_",str(i),".png"]),
        show=False,
    )

    freq, cluster_label = analysis.cluster_freq_by_cat(
        cluster_ids_by_frame, data_obj.id
    )

    lesion_freqs = freq[data_obj.meta["Condition"] == "Lesion"]

    _, r2[i] = analysis.elastic_net_cv(lesion_freqs, SCORES, "".join([paths["out_path"], params["label"], "/test_"]))

import pdb; pdb.set_trace()
plt.plot(sigma_list, r2)
plt.xlabel("Sigma")
plt.ylabel("R2 Score")
plt.savefig("".join([paths["out_path"], params["label"], "/r2_v_sigma.png"]))
plt.close()

plt.plot(sigma_list, np.clip(r2,0,None))
plt.xlabel("Sigma")
plt.ylabel("R2 Score")
plt.savefig("".join([paths["out_path"], params["label"], "/r2_v_sigma_clipped.png"]))
plt.close()