import functools
import numpy as np
import time

from neuroposelib import DataStruct as ds
from typing import Optional, Union, List
# import faiss
import tqdm

# import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from skimage import measure
import pickle


class Embed:
    def __init__(
        self,
        n_neighbors: int = 150,
        embed_method: str = "fitsne",
        transform_method: str = "knn",
        min_dist: float = 0.5,
        spread: float = 1.0,
        n_iter: int = 1000,
        perplexity: Union[str, int] = "auto",
        lr: Union[str, float] = "auto",
        k: int = 5,
        n_trees: int = 100,
        embedder=None,
        template=None,
        temp_embedding=None,
    ):
        self.n_neighbors = n_neighbors

        self.min_dist = min_dist
        self.spread = spread

        self.n_iter = n_iter
        self.perplexity = perplexity
        self.lr = lr
        self.k = k
        self.n_trees = n_trees

        self.embed_method = embed_method

        self.transform_method = transform_method
        self.template = template
        self.temp_embedding = temp_embedding

        self.embedder = embedder

    def embed(
        self,
        features: Optional[np.ndarray] = None,
        n_iter: Optional[int] = None,
        n_neighbors: Optional[int] = None,
        perplexity: Optional[Union[str, int]] = None,
        lr: Optional[Union[str, float]] = None,
        min_dist: Optional[float] = None,
        spread: Optional[float] = None,
        method: Optional[str] = None,
        save_self: bool = True,
    ):
        """
        Calculate t-SNE embedding of template values
        """
        if features is None:
            features = self.template

        self._n = features.shape[0]
        if not n_neighbors:
            n_neighbors = self.n_neighbors
        if not method:
            method = self.embed_method
        if not min_dist:
            min_dist = self.min_dist
        if not spread:
            spread = self.spread
        if not n_iter:
            n_iter = self.n_iter
        if not lr:
            lr = self.lr
        if not perplexity:
            perplexity = self.perplexity

        # if method == "tsne_cuda":
        #     print("Running CUDA tSNE")

        # if lr == "auto":
        #     lr = int(features.shape[0] / 12)

        # if perplexity == "auto":
        #     perplexity = max(int(features.shape[0] / 100), 30)

        # tsne = tc.TSNE(
        #     n_iter=n_iter,
        #     verbose=2,
        #     num_neighbors=n_neighbors,
        #     perplexity=perplexity,
        #     learning_rate=lr,
        # )
        # embed_vals = tsne.fit_transform(features)
        if method == "fitsne":
            print("Running fitsne via openTSNE")
            import openTSNE

            partial_tsne = functools.partial(
                openTSNE.TSNE,
                learning_rate=lr,
                neighbors="annoy",
                negative_gradient_method="fft",
                n_jobs=-1,
                exaggeration=1.5,
                verbose=True,
            )
            if perplexity == "auto":
                tsne = partial_tsne()
            else:
                assert isinstance(perplexity, int)
                tsne = partial_tsne(perplexity=perplexity)
            embed_vals = np.array(
                tsne.fit(features.astype(np.float64)), dtype=features.dtype
            )

        elif method == "umap":
            import umap

            print("Running UMAP")
            embedder = umap.UMAP(
                n_neighbors=n_neighbors, spread=spread, min_dist=min_dist, verbose=True
            )
            embed_vals = embedder.fit_transform(features).astype(features.dtype)
            if save_self:
                self.embedder = embedder
        else:
            raise ValueError(f"Unexpected method {method}")

        if save_self:
            self.template = features
            self.temp_embedding = embed_vals

        return embed_vals

    def predict(
        self,
        data: Union[np.ndarray, ds.DataStruct],
        transform_method: Optional[str] = None,
        n_trees: Optional[int] = None,
        k: Optional[int] = None,
        template: Optional[np.ndarray] = None,
        temp_embedding: Optional[np.ndarray] = None,
    ):
        """
        Uses prediction method to embed points onto template

        IN:
            data - n_frames x n_features
        OUT:
            embed_vals - KNN reembedded values
        """
        if transform_method is None:
            transform_method = self.transform_method
        if n_trees is None:
            n_trees = self.n_trees
        if k is None:
            k = self.k
        if template is None:
            template = self.template
        if temp_embedding is None:
            temp_embedding = self.temp_embedding

        start = time.time()
        if transform_method == "umap":
            print("Predicting using UMAP")
            embed_vals = self.embedder.transform(data)

        elif transform_method == "knn":
            print("Predicting using KNN")
            print(k)
            knn = KNNEmbed(k=k).fit(template, temp_embedding)
            embed_vals = knn.predict_x(data, weights="distance")

        # elif transform_method == "xgboost":
        #     import xgboost as xgb

        #     print("Predicting using XGBoost RF")
        #     print(n_trees)
        #     embed_vals = np.zeros((np.shape(data)[0], 2))
        #     for i in range(2):
        #         embed = xgb.XGBRFRegressor(n_estimators=n_trees, verbosity=2).fit(
        #             template, temp_embedding[:, i]
        #         )
        #         embed_vals[:, i] = embed.predict(data)

        elif transform_method == "sklearn_rf":
            from sklearn.ensemble import RandomForestRegressor

            rf_embed = RandomForestRegressor(n_estimators=n_trees, n_jobs=-1)
            rf_embed = rf_embed.fit(template, temp_embedding)
            embed_vals = rf_embed.predict(data)

        print("Total Time embedding: ", time.time() - start)
        return embed_vals


class BatchEmbed(Embed):
    def __init__(
        self,
        sampling_n: int = 20,
        n_neighbors: int = 150,
        sigma: int = 15,
        batch_method: str = "fitsne",
        embed_method: str = "fitsne",
        transform_method: str = "knn",
        min_dist: float = 0.5,
        spread: float = 1.0,
        n_iter: int = 1000,
        perplexity: Union[str, int] = "auto",
        lr: Union[str, int] = "auto",
        k: int = 5,
        n_trees: int = 100,
        embedder=None,
        template=None,
        temp_idx=[],
        temp_embedding=None,
    ):
        """
        t-SNE parameters here are used in the embedding of batches,
        not for the final template itself
        """
        super().__init__(
            n_neighbors=n_neighbors,
            embed_method=embed_method,
            transform_method=transform_method,
            min_dist=min_dist,
            spread=spread,
            n_iter=n_iter,
            perplexity=perplexity,
            lr=lr,
            k=k,
            n_trees=n_trees,
            embedder=embedder,
            template=template,
            temp_embedding=temp_embedding,
        )
        self.sampling_n = sampling_n
        self.sigma = sigma
        self.batch_method = batch_method
        self.temp_idx = temp_idx

    def fit(
        self,
        data: Union[np.ndarray, ds.DataStruct],
        batch_id: Optional[Union[np.ndarray, List[Union[int, str]]]] = None,
        # save_batchmaps: Optional[str] = None,
        embed_temp: bool = True,
    ):
        """ """
        # if save_batchmaps:
        #     import visualization as vis

        #     save_path = "".join([save_batchmaps, "/batch_maps/"])
        #     if not os.path.exists(save_path):
        #         os.makedirs(save_path)

        #     filename = "".join([save_path, self.embed_method])

        self.template = np.empty((0, data.shape[1]))
        self.temp_idx = []
        for batch in tqdm.tqdm(np.unique(batch_id)):
            data_by_ID = data[batch_id == batch, :]  # Subsetting data by batch

            embed_vals = self.embed(
                data_by_ID, method=self.batch_method, save_self=False
            )

            ws = Watershed(
                sigma=self.sigma, n_bins=1000, max_clip=1, log_out=True, pad_factor=0
            )
            cluster_labels = ws.fit_predict(embed_vals)

            # if save_batchmaps:
            #     ws.plot_density(
            #         filepath="".join([filename, str(batch), "_density.png"]),
            #         watershed=True,
            #     )
            #     vis.scatter(
            #         embed_vals, filepath="".join([filename, str(batch), "_scatter.png"])
            #     )

            sampled_points, idx = self.__sample_clusters(
                data_by_ID, cluster_labels, sample_size=self.sampling_n
            )

            idx = np.nonzero(batch_id == batch)[0][idx]
            self.template = np.append(self.template, sampled_points, axis=0)
            self.temp_idx += list(idx)

        if embed_temp:
            self.embed(self.template, save_self=True)  # template would get saved twice

        return self

    def fit_predict(
        self,
        data: Union[np.ndarray, ds.DataStruct],
        batch_id: Optional[Union[np.ndarray, List[Union[int, str]]]] = None,
        save_batchmaps: Optional[str] = None,
    ):
        self.fit(
            data=data, batch_id=batch_id, save_batchmaps=save_batchmaps, embed_temp=True
        )
        embed_vals = self.predict(data)

        return embed_vals

    def __sample_clusters(
        self,
        data,
        meta_name: Union[np.ndarray, List[Union[int, str]]],
        sample_size: int = 20,
    ):
        """
        Equally sampling points from
        IN:
            data - All of the data in dataset (may be downsampled)
            meta_name - Cluster number for each point in `data`
            sample_size - Number of points to sample from a cluster
        OUT:
            sampled_points - Values of sampled points from `data`
            idx - Index in `data` of sampled points
        """
        data = np.append(
            data, np.expand_dims(np.arange(np.shape(data)[0]), axis=1), axis=1
        )
        sampled_points = np.empty((0, np.shape(data)[1]))
        for meta_id in np.unique(meta_name):
            points = data[meta_name == meta_id, :]
            if len(points) < sample_size:
                # If fewer points, just skip (probably artifactual cluster)
                continue
                # sampled_idx = np.random.choice(np.arange(len(points)), size=size, replace=True)
                # sampled_points = np.append(sampled_points, points[sampled_idx,:], axis=0)
            else:
                num_points = min(len(points), sample_size)
                sampled_points = np.append(
                    sampled_points, np.random.permutation(points)[:num_points], axis=0
                )
        print("Number of points sampled")
        print(sampled_points.shape)
        return (
            sampled_points[:, :-1],
            np.squeeze(sampled_points[:, -1]).astype(int).tolist(),
        )

    def save_pickle(self, filepath: str = "./plot_folder/"):
        pickle.dump(self, open("".join([filepath, "batch_embed.p"]), "wb"))
        return self

    def load_pickle(self, filepath: str = "./plot_folder/batch_embed.p"):
        self = pickle.load(open(filepath, "rb"))
        return self


# class KNNGraph:
#     """
#     Using faiss to run k-Nearest Neighbors algorithm
#     """

#     def __init__(self, k: int = 5):
#         """
#         Creates data structure for fast search of neighbors
#         IN:
#             X - Features of training data
#             y - Training data
#         """
#         self.k = k

#     def fit(self, X):
#         self.index = faiss.IndexFlatL2(X.shape[1])
#         self.index.add(np.ascontiguousarray(X, dtype=np.float32))

#         return self


# class KNNEmbed(KNNGraph):
#     """
#     Using faiss to run k-Nearest Neighbors algorithm for embedding of points in 2D
#     when given high-D data and low-D embedding of template data
#     """

#     def __init__(self, k: int = 5):
#         super().init(k)
#         self.distances = None
#         self.indices = None

#     def predict_x(self, X, y, weights="standard"):
#         """
#         Predicts embedding of data using KNN
#         IN:
#             X - Features of data to predict
#             weights - 'standard' or 'distance' determines weights on nearest neighbors
#         OUT:
#             predictions - output predictions
#         """
#         print("Predicting")
#         distances, indices = self.index.search(
#             np.ascontiguousarray(X, dtype=np.float32), k=self.k
#         )
#         y = np.ascontiguousarray(y, dtype=np.float32)
#         votes = self.y[indices]

#         if weights == "distance":
#             min_dist = np.min(distances[np.nonzero(distances)]) / 2
#             distances = np.clip(distances, min_dist, None)
#             weights = 1 / distances
#             weights = weights / np.repeat(
#                 np.expand_dims(np.sum(weights, axis=1), axis=1), self.k, axis=1
#             )
#         else:
#             weights = 1 / self.k

#         weights = np.repeat(np.expand_dims(weights, axis=2), 2, axis=2)
#         predictions = np.sum(votes * weights, axis=1)
#         return predictions


class GaussDensity:
    """
    Class for creating Gaussian density maps of 2D scatter data
    """

    def __init__(
        self,
        sigma: int = 15,
        n_bins: int = 1000,
        max_clip: float = 0.75,
        log_out: bool = False,
        pad_factor: float = 0.025,
    ):
        self.sigma = sigma
        self.n_bins = n_bins
        self.max_clip = max_clip
        self.log_out = log_out
        self.pad_factor = pad_factor

        self.hist_range = None

        # TODO: More consideration for when these save
        self.density = None
        self.data_in_bin = None

    def hist(self, data: np.ndarray, new: bool = True):
        """
        Run 2D histogram

        IN:
            data - Data to convert to density map (n_frames x 2)
            new - Map onto old hist range and bins if False
        OUT:
            hist - Calculated 2d histogram  (n_bins x n_bins)
        """
        range_len = (
            np.ceil(np.amax(data, axis=0)) - np.floor(np.amin(data, axis=0))
        ).astype(int)
        padding = (range_len * self.pad_factor).astype(data.dtype)

        # Calculate x and y limits for histogram and density
        if new or (self.hist_range is None):
            print("Calculating new histogram ranges")
            self.hist_range = [
                [np.amin(data[:, 0]) - padding[0], np.amax(data[:, 0]) + padding[0]],
                [np.amin(data[:, 1]) - padding[1], np.amax(data[:, 1]) + padding[1]],
            ]

        hist, self.xedges, self.yedges = np.histogram2d(
            data[:, 0],
            data[:, 1],
            bins=[self.n_bins, self.n_bins],
            range=self.hist_range,
            density=True,
        )
        hist = np.rot90(hist)

        assert (self.xedges[0] < self.xedges[-1]) and (self.yedges[0] < self.yedges[1])

        return hist

    def fit_density(self, data: np.ndarray, new: bool = True, map_bin: bool = True):
        """
        Calculate Gaussian density for 2D embedding

        IN:
            data - Data to convert to density map (n_frames x 2)
            new - Map onto old hist range and bins if False
        OUT:
            density - Calculated density map (n_bins x n_bins)
        """
        # 2D histogram
        hist = self.hist(data, new)

        # Calculates density using gaussian filter
        density = gaussian_filter(hist, sigma=self.sigma)
        if self.log_out:
            density = np.log1p(density)
        density = np.clip(
            density, None, np.amax(density) * self.max_clip
        )  # clips max for better visualization of clusters

        if map_bin:
            # Maps each data point to bin indices and saves to self
            # May need some more consideration for when this saves and doesn't save
            self.data_in_bin = self.map_bins(data)

        if new:
            self.density = density

        return density

    def map_bins(self, data: np.ndarray):
        """
        Find which bin in histogram/density map each data point is a part of
        IN:
            edges: self.xedges and self.yedges must be calculated from np.histogram (represents edge values of bins)
            data: Data to be transformed
        OUT:
            data_in_bin: Indices (returns n_frames x 2) of data in density map (shape n_bins x n_bins)
        """
        if self.xedges is None:
            print("Could not find histogram, computing now")
            self.density = None
            self.hist(data, new=True)

        dtype = np.int32 if data.dtype == np.float32 else int

        data_in_bin = np.zeros(np.shape(data), dtype)

        # This is actually slower
        # data_in_bin[:,1] = np.argmax(self.xedges>np.repeat(data[:,0][:,None],len(self.xedges),axis=1),axis=1)-1
        # data_in_bin[:,0] = self.n_bins-np.argmax(self.yedges>np.repeat(data[:,1][:,None],len(self.yedges),axis=1),axis=1)-1

        for i in range(data_in_bin.shape[0]):
            data_in_bin[i, 1] = (
                np.argmax(self.xedges > data[i, 0]) - 1
            )  # ,0,self.n_bins-1
            data_in_bin[i, 0] = (
                self.n_bins - np.argmax(self.yedges > data[i, 1]) - 1
            )  # ,0,self.n_bins-1)

        return data_in_bin

    # def plot_density(self, filepath: str = "./plot_folder/density.png"):
    #     f = plt.figure()
    #     ax = f.add_subplot(111)
    #     ax.imshow(self.density)
    #     ax.set_aspect("auto")
    #     plt.savefig(filepath, dpi=400)
    #     plt.close()


class Watershed(GaussDensity):

    def __init__(
        self,
        sigma: int = 15,
        n_bins: int = 1000,
        max_clip: float = 0.75,
        log_out: bool = False,
        pad_factor: float = 0.025,
        density_thresh: float = 17,
    ):
        super().__init__(
            sigma=sigma,
            n_bins=n_bins,
            max_clip=max_clip,
            log_out=log_out,
            pad_factor=pad_factor,
        )
        self.density_thresh = density_thresh / (n_bins**2)

        self.watershed_map = None
        self.borders = None

        self.density = None  # TODO: Consider more when this saves and doesn't

    def fit(self, data: np.ndarray):
        """
        Running watershed clustering on data
        IN:
            data - ds.DataStruct object or numpy array (frames x 2) of t-SNE coordinates
        OUT:
            self.density
        """
        from skimage.segmentation import watershed

        self.density = self.fit_density(data, new=True, map_bin=False)

        print("Calculating watershed")
        self.watershed_map = watershed(
            -self.density, mask=self.density > self.density_thresh, watershed_line=False
        )
        self.watershed_map[self.density < self.density_thresh] = 0
        self.borders = np.empty((0, 2), dtype=data.dtype)

        for i in range(1, len(np.unique(self.watershed_map))):
            contour = measure.find_contours(self.watershed_map.T == i, 0.5)[0]
            self.borders = np.append(self.borders, contour, axis=0)

        return self

    def predict(self, data: Optional[Union[ds.DataStruct, np.ndarray]] = None):
        """
        Predicts the cluster label of data

        Requires knowledge of what bin data is in in the histogram/density map

        IN:
            data - XY coordinates of data to be predicted
        OUT:
            cluster_labels - cluster labels of all data
        """
        dtype = np.int32 if data.dtype == np.float32 else int
        data_in_bin = self.map_bins(data)

        cluster_labels = self.watershed_map[
            data_in_bin[:, 0].astype(dtype), data_in_bin[:, 1].astype(dtype)
        ]
        print(str(int(np.amax(cluster_labels) + 1)), "clusters detected")
        print(str(np.unique(cluster_labels).shape), "unique clusters detected")
        print(np.unique(cluster_labels))

        return cluster_labels

    def fit_predict(self, data: Optional[Union[ds.DataStruct, np.ndarray]] = None):
        self.fit(data)
        cluster_labels = self.predict(data)
        return cluster_labels

    # def plot_watershed(
    #     self, filepath: str = "./plot_folder/watershed.png", borders: bool = True
    # ):
    #     f = plt.figure()
    #     ax = f.add_subplot(111)
    #     ax.imshow(self.watershed_map)
    #     ax.set_aspect("auto")
    #     if borders:
    #         ax.plot(self.borders[:, 0], self.borders[:, 1], ".r", markersize=0.05)
    #     plt.savefig("".join([filepath, "_watershed.png"]), dpi=400)
    #     plt.close()

    # def plot_density(
    #     self, filepath: str = "./plot_folder/density.png", watershed: bool = True
    # ):
    #     f = plt.figure()
    #     ax = f.add_subplot(111)
    #     if watershed:
    #         ax.plot(self.borders[:, 0], self.borders[:, 1], ".r", markersize=0.1)
    #     ax.imshow(self.density)
    #     ax.set_aspect("auto")
    #     plt.savefig(filepath, dpi=400)
    #     plt.close()


# class KFoldEmbed:
#     def __init__(
#         k_split: int = 10,
#         param_range=list(range(1, 22, 2)),
#         plot_folder: str = "./plots/",
#         watershed: bool = True,
#     ):
#         self.k_split = k_split
#         self.plot_folder = plot_folder
#         self.param = param

#         self.mse = []
#         self.euc = []

#     def run(
#         self,
#         embedder: Union[BatchEmbed, Embed],
#         param: str,
#     ):
#         """
#         param can be either k, n_tree, or
#         """
#         from sklearn.model_selection import KFold

#         print("Embedding 10-fold data")
#         template = embedder.template
#         temp_embedding = embedder.temp_embedding
#         print("Template shape: ", template.shape)
#         print("Predictions shape: ", temp_embedding.shape)
#         kf = KFold(n_splits=k_split, shuffle=True)
#         preds_max_dist = np.sqrt(
#             np.sum(
#                 (np.amax(temp_embedding, axis=0) - np.amin(temp_embedding, axis=0)) ** 2
#             )
#         )

#         metric_vals, min_metric_embedding = [], []

#         for param_val in param_range:
#             setattr(embedder, param, param_val)  # seting new param
#             print("Reembedding with ", param_val, " ", param)
#             kf_embedding = np.empty((0, 2))
#             start = time.time()
#             mse_k, euc_k = np.zeros(shape)

#             for train, test in tqdm.tqdm(kf.split(template, temp_embedding)):
#                 # Embed the 90
#                 kf_temp_embedding = embedder.embed(
#                     data=template[train], save_self=False
#                 )

#                 # Reembed the 10 using the 90
#                 kf_embedding = embedder.predict(
#                     data=template[test],
#                     template=template[train],
#                     temp_embedding=kf_temp_embedding,
#                 )

#                 kf_embedding = np.append(kf_embedding, reembedding, axis=0)
#                 test_idx += test

#                 mse_k[test] = (temp_embedding[test] - kf_embedding) ** 2

#             print("Total Time K-Fold Reembedding: ", time.time() - start)

#             euc = np.mean(
#                 np.sqrt(
#                     np.sum(
#                         (data_shuffled[: kf_embedding.shape[0], :2] - kf_embedding)
#                         ** 2,
#                         axis=1,
#                     )
#                 )
#             )
#             mse = np.mean(
#                 np.sum(
#                     (data_shuffled[: kf_embedding.shape[0], :2] - kf_embedding) ** 2,
#                     axis=1,
#                 )
#             )

#             curr_metric = curr_metric / max_dist
#             print(curr_metric)

#             print("Reembedding Metric: ", curr_metric)

#             # if metric is empty or curr_metric is lowest so far
#             if not metric_vals or all(curr_metric < val for val in metric_vals):
#                 min_metric_embedding = kf_embedding
#                 min_metric_nn = nn

#             metric_vals += [curr_metric]

#         ws = Watershed(sigma=15, max_clip=1, log_out=True, pad_factor=0.05)
#         cluster_true = ws.fit_predict(temp_embedding)

#         cluster_preds = ws.predict(reembedding)

#         f = plt.figure()
#         plt.scatter(
#             predictions[:, 0],
#             predictions[:, 1],
#             marker=".",
#             s=3,
#             linewidths=0,
#             c="b",
#             label="Targets",
#         )
#         plt.scatter(
#             min_metric_embedding[:, 0],
#             min_metric_embedding[:, 1],
#             marker=".",
#             s=3,
#             linewidths=0,
#             c="m",
#             label="CV Predictions",
#         )
#         plt.legend()
#         plt.savefig(
#             "".join([plot_folder, "k_fold_mbed_", str(min_metric_nn), "nn.png"]),
#             dpi=400,
#         )
#         plt.close()

#         f = plt.figure()
#         plt.plot(nn_range, metric_vals, marker="o", c="k")
#         plt.savefig("".join([plot_folder, "k_fold_embed_metric.png"]), dpi=400)
#         plt.close()

#         return min_metric_nn
