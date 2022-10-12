# import DataStruct as ds
# from sklearn import metrics
# import pickle
# import numpy as np
# import matplotlib.pyplot as plt
# from embed import Watershed
# import tqdm

# paths_dict = {
#     'batch_fitsne_knn': '../results/24_final/batch_fitsne_4_25/datastruct.p',
#     'batch_fitsne_rf': '../results/24_final/batch_fitsne_425_rf/datastruct.p',
#     'batch_umap': '../results/24_final/fitsne_425/datastruct.p',
#     'fitsne': '../results/24_final/batch_fitsne_umap_425/datastruct.p',
# }
# sigma_list = [7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,30] #5,6,7,8,9,
# silhouette = np.zeros((len(paths_dict.keys()), len(sigma_list)))
# n_clusters = np.zeros((len(paths_dict.keys()), len(sigma_list)))
# ch_index = np.zeros((len(paths_dict.keys()), len(sigma_list)))
# db_index = np.zeros((len(paths_dict.keys()), len(sigma_list)))

# for i,key in tqdm.tqdm(enumerate(paths_dict)):
#     ds = pickle.load(open(paths_dict[key],'rb'))
#     # ds.cluster_freq('AnimalID', True)
#     for j,sigma in enumerate(sigma_list):
#         ws = Watershed(sigma = sigma,
#                         max_clip = 1,
#                         log_out = True,
#                         pad_factor = 0.05)
#         cluster_labels = ws.fit_predict(data = ds.embed_vals)
#         # print("Calculating Silhouette")
#         # silhouette[i,j] = metrics.silhouette_score(ds.embed_vals, cluster_labels)
#         n_clusters[i,j] = np.max(cluster_labels)+1
#         print("Calculating CH Index")
#         ch_index[i,j] = metrics.calinski_harabasz_score(ds.embed_vals, cluster_labels)
#         print(ch_index)
#         # print("Calculating DB Index")
#         # db_index[i,j] = metrics.davies_bouldin_score(ds.embed_vals, cluster_labels)
#         # print(db_index)

# f,ax_arr = plt.subplots(1,2,figsize=(12,6))

# for i, key in enumerate(paths_dict):
#     ax_arr[0].plot(sigma_list, n_clusters[i,:], marker='o', label=key)

# ax_arr[0].set_xlabel('Gaussian Kernel Width ($\sigma$)')
# ax_arr[0].set_ylabel('# Clusters')

# # for i, key in enumerate(paths_dict):
# #     ax_arr[0,1].plot(sigma, silhouette[i,:], label=key)

# # ax_arr[0,1].legend()
# # ax_arr[0,1].set_xlabel('Gaussian Kernel Width ($\sigma$)')
# # ax_arr[0,1].set_ylabel('Silhouette Coefficient')

# for i, key in enumerate(paths_dict):
#     ax_arr[1].plot(sigma_list, ch_index[i,:],marker='o', label=key)

# ax_arr[1].set_xlabel('Gaussian Kernel Width ($\sigma$)')
# ax_arr[1].set_ylabel('Calinski-Harabasz Index')

# # for i, key in enumerate(paths_dict):
# #     ax_arr[2].plot(sigma_list, db_index[i,:], label=key)

# # ax_arr[2].legend()
# # ax_arr[2].set_xlabel('Gaussian Kernel Width ($\sigma$)')
# # ax_arr[2].set_ylabel('Davies-Bouldin Index')
# f.tight_layout()
# lines, labels = ax_arr[0].get_legend_handles_labels()
# f.legend(lines, labels, loc='upper center', ncol=4)
# plt.savefig('../results/24_final/indexes.png',dpi=400)
