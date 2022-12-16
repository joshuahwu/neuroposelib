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

scores = np.array(
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

# vis.density_cat(
#     data=data_obj,
#     column="Condition",
#     watershed=data_obj.ws,
#     n_col=4,
#     filepath="".join(
#         [paths["out_path"], params["label"], "/density_Condition.png"]
#     ),
# )

# import pdb; pdb.set_trace()

# vis.skeleton_vid3D_cat(data_obj, 'Cluster', n_skeletons=1, filepath = ''.join([paths['out_path'],params['label'],'/single/']))
# freq, cluster_label = analysis.cluster_freq_by_cat(
#     data_obj.data["Cluster"], data_obj.id
# )

# embed = Embed(
#     embed_method=params["single_embed"]["method"],
#     perplexity=15,
#     lr=params["single_embed"]["lr"],
# )
# mice_embed = embed.embed(freq, save_self=True)

# vis.scatter_by_cat(
#     mice_embed,
#     data_obj.meta["Condition"].values,
#     label="Condition",
#     filepath="".join([paths["out_path"], params["label"], "/"]),
#     color = ["#00b7c7", "#dc0ab4"],
#     marker_size = 200
# )

# f = plt.figure()
# ax = f.add_subplot(111)
# ax.scatter(
#         mice_embed[data_obj.meta["Condition"].values=="Baseline",0],
#         mice_embed[data_obj.meta["Condition"].values=="Baseline",1],
#         marker=".",
#         s=200,
#         linewidths=0,
#         c="tab:gray",
#         label = "Baseline",
#         alpha=1
#     )
# ax.legend()
# plot = ax.scatter(
#         mice_embed[data_obj.meta["Condition"].values=="Lesion",0],
#         mice_embed[data_obj.meta["Condition"].values=="Lesion",1],
#         marker=".",
#         s=200,
#         linewidths=0,
#         c = scores,
#         label = "Lesion",
#         cmap=sns.color_palette("crest", as_cmap=True),
#         alpha=1
#     )
# ax.set_box_aspect(0.8)
# ax.set_xlabel("t-SNE 1"); ax.set_ylabel("t-SNE 2")
# plt.colorbar(plot, ax=ax, fraction = 0.047*0.8)
# plt.savefig("".join([paths["out_path"], params["label"], "/scatter_fluorescence.png"]))
# plt.close()

# # vis.scatter(
# #     mice_embed,
# #     scores_concat,
# #     marker_size = 20,
# #     filepath = "".join([paths["out_path"], params["label"], "/scatter_fluorescence.png"]),
# # )

# vis.scatter_by_cat(
#     mice_embed,
#     data_obj.meta["Sex"].values,
#     label="Sex",
#     filepath="".join([paths["out_path"], params["label"], "/"]),
#     marker_size = 200
# )

# import pdb; pdb.set_trace()

vis.cluster_freq_cond(data_obj, cat1 = 'Condition', cat2 = 'AnimalID', filepath=''.join([paths['out_path'],params['label'],'/']))

# pairwise_analysis = np.delete(freq, 58, axis=0)
# analysis.pairwise_cosine(freq, "".join([paths["out_path"], params["label"], "/"]))

# lesion_freqs = freq[data_obj.meta["Condition"] == "Lesion"]
# healthy_freqs = freq[data_obj.meta["Condition"] == "Baseline"]
# paired_cos = analysis.cosine_similarity(
#     freq[data_obj.meta["Condition"] == "Baseline"],
#     lesion_freqs,
# )
# plt.scatter(scores, paired_cos)
# plt.xlabel("Integrated Fluorescence %")
# plt.ylabel("Healthy v Lesion Cosine Similarity")
# plt.savefig("".join([paths["out_path"], params["label"], "/fluor_cos_sim.png"]))
# plt.close()

# # sort_idx = np.argsort(scores)
# # scores_sorted = scores[sort_idx]
# # freq_sorted = lesion_freqs[sort_idx,:]
# # healthy_means = freq[data_obj.meta["Condition"] == "Baseline"].mean(axis=0)
# # health_sort_idx = np.argpartition(healthy_means,-30)[-30:]
# # norm_freqs = lesion_freqs[:,health_sort_idx]/healthy_means[health_sort_idx]

# analysis.lstsq(lesion_freqs, scores, "".join([paths["out_path"], params["label"], "/"]))
# analysis.elastic_net(lesion_freqs, scores, "".join([paths["out_path"], params["label"], "/"]))
# analysis.random_forest(lesion_freqs, scores, "".join([paths["out_path"], params["label"], "/"]))

# analysis.elastic_net(freq[data_obj.meta["Condition"] == "Baseline"], scores, "".join([paths["out_path"], params["label"], "/healthy_"]))

# norm_freqs = lesion_freqs - freq[data_obj.meta["Condition"]=="Baseline"]
# analysis.lstsq(norm_freqs, scores, "".join([paths["out_path"], params["label"], "/healthy_subtracted_"]))
# analysis.elastic_net(norm_freqs, scores, "".join([paths["out_path"], params["label"], "/healthy_subtracted_"]))


# regr = ElasticNet(alpha=0.1, l1_ratio=0.7)

# regr.fit(StandardScaler().fit_transform(lesion_freqs), scores)
# pred_y = regr.predict(StandardScaler().fit_transform(lesion_freqs))
# plt.scatter(scores, pred_y, s=30)
# plt.savefig("elastic_test.png")
# coeffs = regr.coef_
# biggest = np.argpartition(coeffs, -5)[-5:]
# import pdb; pdb.set_trace()

# sns.set(rc={'figure.figsize':(12,10)})
# for i in range(lesion_freqs.shape[1]):
#     plt.scatter(scores,lesion_freqs[:,i], s=30)
#     plt.xlabel("Fluorescence")
#     plt.ylabel("Cluster Occupancy")
#     plt.savefig("".join([paths["out_path"], params["label"], "/cluster_fluor/cluster_", str(i),".png"]))
#     plt.close()

analysis.elastic_net_cv(lesion_freqs, scores, "".join([paths["out_path"], params["label"], "/test_"]))
# analysis.elastic_net(freq[data_obj.meta["Condition"] == "Baseline"], scores, "".join([paths["out_path"], params["label"], "/healthy_log_"]))
# analysis.elastic_net(lesion_freqs, scores, "".join([paths["out_path"], params["label"], "/healthy_log_"]))
# print("All Done")