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
import pandas as pd

analysis_key = "histology_no24"
paths = read.config("../configs/path_configs/" + analysis_key + ".yaml")
params = read.config("../configs/param_configs/fitsne.yaml")

scores = pd.read_excel("./histology_vals_no24.xlsx")

score_cats = ["DMS_mean","DLS_mean", "VS_mean", "TS_mean",
              "DMS_sum","DLS_sum","VS_sum","TS_sum"]

# data_obj = pickle.load(
#     open("/home/exx/Desktop/GitHub/results/histology_no24/fitsne/datastruct.p", "rb")
# )

data_obj = pickle.load(
    open("".join([paths["out_path"], params["label"], "/datastruct.p"]), "rb")
)

# # Watershed clustering
# data_obj.ws = Watershed(
#     20, max_clip=1, log_out=True, pad_factor=0.05
# )
# data_obj.data.loc[:, "Cluster"] = data_obj.ws.fit_predict(data=data_obj.embed_vals)

# print("Writing Data Object to pickle")
# data_obj.write_pickle("".join([paths["out_path"], params["label"], "/"]))


# ## Plotting 
# vis.density_cat(
#     data=data_obj,
#     column="Condition",
#     watershed=data_obj.ws,
#     n_col=4,
#     filepath="".join(
#         [paths["out_path"], params["label"], "/density_Condition.png"]
#     ),
# )

# ## Plot skeletons for each cluster
# # vis.skeleton_vid3D_cat(data_obj, 'Cluster', n_skeletons=1, filepath = ''.join([paths['out_path'],params['label'],'/single/']))

## Calculate cluster frequencies
freq, cluster_label = analysis.cluster_freq_by_cat(
    data_obj.data["Cluster"], data_obj.id
)

# ## t-SNE of cluster occupancy vectors
# embed = Embed(
#     embed_method=params["single_embed"]["method"],
#     perplexity=15,
#     lr=params["single_embed"]["lr"],
# )
# mice_embed = embed.embed(freq, save_self=True)

# ## t-SNE plot by condition
# vis.scatter_by_cat(
#     mice_embed,
#     data_obj.meta["Condition"].values,
#     label="Condition",
#     filepath="".join([paths["out_path"], params["label"], "/"]),
#     color = ["#00b7c7", "#dc0ab4"],
#     size = 200
# )


# ## t-SNE plots by fluorescence scores
# f = plt.figure(figsize=(20,10))
# for i, cat in enumerate(score_cats):
#     ax = f.add_subplot(240+i+1)
#     ax.scatter(
#             mice_embed[data_obj.meta["Condition"].values=="Baseline",0],
#             mice_embed[data_obj.meta["Condition"].values=="Baseline",1],
#             marker=".",
#             s=200,
#             linewidths=0,
#             c="tab:gray",
#             label = "Baseline",
#             alpha=1
#         )
#     ax.legend()
#     plot = ax.scatter(
#             mice_embed[data_obj.meta["Condition"].values=="Lesion",0],
#             mice_embed[data_obj.meta["Condition"].values=="Lesion",1],
#             marker=".",
#             s=200,
#             linewidths=0,
#             c = scores[cat].values,
#             label = "Lesion",
#             cmap=sns.color_palette("crest", as_cmap=True),
#             alpha=1
#         )
#     ax.set_box_aspect(0.8)
#     ax.set_xlabel("t-SNE 1"); ax.set_ylabel("t-SNE 2")
#     ax.set_title(cat)

#     plt.colorbar(plot, ax=ax, fraction = 0.047*0.8)
# plt.tight_layout()
# plt.savefig("".join([paths["out_path"], params["label"], "/scatter_fluorescence.png"]))
# plt.close()

# ## t-SNE by animal sex
# vis.scatter_by_cat(
#     mice_embed,
#     data_obj.meta["Sex"].values,
#     label="Sex",
#     filepath="".join([paths["out_path"], params["label"], "/"]),
#     size = 200
# )

## Visualize plot of cluster occupancies for each animal and each condition
vis.cluster_freq_cond(data_obj, cat1 = 'Condition', cat2 = 'AnimalID', filepath=''.join([paths['out_path'],params['label'],'/']))

# pairwise_analysis = np.delete(freq, 58, axis=0) # remove outlier animal not actually sure if this is right

## Average cosine similarity within a condition
analysis.pairwise_cosine(freq, "".join([paths["out_path"], params["label"], "/"]))

## Separating lesion and healthy cluster occupancies
lesion_freqs = freq[data_obj.meta["Condition"] == "Lesion"]
healthy_freqs = freq[data_obj.meta["Condition"] == "Baseline"]

# Out-of-condition paired cosine similarities
paired_cos = analysis.cosine_similarity(
    freq[data_obj.meta["Condition"] == "Baseline"],
    lesion_freqs,
)

# Plot paired cosine similarity as a function of fluorescence %
f = plt.figure(figsize=(20,10))
for i, cat in enumerate(score_cats):
    ax = f.add_subplot(240+i+1)
    ax.scatter(scores[cat], paired_cos)
    ax.set_xlabel("Integrated Fluorescence %")
    ax.set_ylabel("Healthy v Lesion Cosine Similarity")
    ax.set_title(cat)
plt.savefig("".join([paths["out_path"], params["label"], "/fluor_cos_sim.png"]))
plt.close()

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

# lesion_freqs = np.delete(lesion_freqs, 58, axis=0) # remove outlier animal not actually sure if this is right
sns.set(rc={'figure.figsize':(6,5)})
f = plt.figure(figsize=(20,10))
for i, cat in enumerate(score_cats):
    scores_y = scores[cat].values
    pred_y, r2 = analysis.elastic_net_cv(lesion_freqs, scores_y, "".join([paths["out_path"], params["label"], "/",cat,"_"]))
    ax = f.add_subplot(240+i+1)

    ax.plot(np.linspace(scores_y.min(), scores_y.max(), 100), np.linspace(scores_y.min(),scores_y.max(),100), markersize=0, color='k', label="y = x")
    ax.legend(loc="upper center")
    ax.scatter(scores_y, pred_y, s=30)
    ax.set_xlabel("Real Fluorescence")
    ax.set_ylabel("Predicted Fluorescence")
    ax.set_title(cat + " R2 = " + str(r2))
plt.savefig("".join([paths["out_path"], params["label"], "/elastic_no.png"]))
plt.close()


# analysis.elastic_net(freq[data_obj.meta["Condition"] == "Baseline"], scores, "".join([paths["out_path"], params["label"], "/healthy_log_"]))
# analysis.elastic_net(lesion_freqs, scores, "".join([paths["out_path"], params["label"], "/healthy_log_"]))
# print("All Done")