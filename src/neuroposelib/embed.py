import functools
import numpy as np
import time
from neuroposelib import DataStruct as ds
from typing import Optional, Union, List, Tuple, Any
import numpy.typing as npt
import tqdm
from scipy.ndimage import gaussian_filter
from skimage import measure

import pickle


class Embed:
    """
    Base class to compute and apply low-dimensional embeddings (t-SNE / UMAP).

    Parameters
    ----------
    n_neighbors : int
        Default number of neighbors for neighbor-based embedders.
    embed_method : str
        Embedding algorithm to use for template construction (e.g. "fitsne", "umap").
    transform_method : str
        Method used to transform new data into the template embedding (e.g. "knn", "umap").
    min_dist : float
        UMAP `min_dist` parameter (only used when `embed_method="umap"`).
    spread : float
        UMAP `spread` parameter (only used when `embed_method="umap"`).
    n_iter : int
        Number of iterations for iterative embedders.
    perplexity : int or str
        t-SNE perplexity. Accepts "auto" or an int.
    lr : float or str
        Learning rate for t-SNE. Accepts "auto" or a float.
    k : int
        Default k for KNN-based transforms.
    n_trees : int
        Default number of trees for ensemble regressors (e.g. random forest).
    embedder : object or None
        If an external embedder object (UMAP etc.) is available, it can be stored here.
    template : array-like, shape (n_template_samples, n_features) or None
        Template high-dimensional data used to build the low-dimensional template.
    temp_embedding : array-like, shape (n_template_samples, n_components) or None
        Low-dimensional embedding of `template`.
    """

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
        embedder: Any = None,
        template: Optional[npt.ArrayLike] = None,
        temp_embedding: Optional[npt.ArrayLike] = None,
    ) -> None:
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
        features: Optional[npt.ArrayLike] = None,
        n_iter: Optional[int] = None,
        n_neighbors: Optional[int] = None,
        perplexity: Optional[Union[str, int]] = None,
        lr: Optional[Union[str, float]] = None,
        min_dist: Optional[float] = None,
        spread: Optional[float] = None,
        method: Optional[str] = None,
        save_self: bool = True,
    ) -> npt.NDArray[Any]:
        """
        Calculate a low-dimensional embedding of `features` using the configured
        embedding method.

        Parameters
        ----------
        features : array-like, shape (n_samples, n_features), optional
            Feature matrix to embed. If None, `self.template` will be used.
        n_iter : int, optional
            Number of iterations (overrides instance default if provided).
        n_neighbors : int, optional
            Number of neighbors (overrides instance default if provided).
        perplexity : int or str, optional
            t-SNE perplexity override ("auto" or int).
        lr : float or str, optional
            Learning rate override ("auto" or float).
        min_dist : float, optional
            UMAP min_dist override.
        spread : float, optional
            UMAP spread override.
        method : str, optional
            Embedding method to use for this call. If None, instance `embed_method`
            is used. Supported: "fitsne", "umap".
        save_self : bool, default True
            If True, save the template and resulting embedding to the instance.

        Returns
        -------
        embed_vals : ndarray, shape (n_samples, n_components)
            2D embedding of the input features.
        """
        if features is None:
            if self.template is None:
                raise ValueError("No features provided and self.template is None.")
            features = self.template

        self._n = np.asarray(features).shape[0]
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
                tsne.fit(np.asarray(features, dtype=np.float64)), dtype=np.asarray(features).dtype
            )

        elif method == "umap":
            import umap

            print("Running UMAP")
            embedder = umap.UMAP(
                n_neighbors=n_neighbors, spread=spread, min_dist=min_dist, verbose=True
            )
            embed_vals = embedder.fit_transform(np.asarray(features)).astype(np.asarray(features).dtype)
            if save_self:
                self.embedder = embedder
        else:
            raise ValueError(f"Unexpected method {method}")

        if save_self:
            self.template = np.asarray(features)
            self.temp_embedding = np.asarray(embed_vals)

        return np.asarray(embed_vals)

    def predict(
        self,
        data: Union[npt.ArrayLike, ds.DataStruct],
        transform_method: Optional[str] = None,
        n_trees: Optional[int] = None,
        k: Optional[int] = None,
        template: Optional[npt.ArrayLike] = None,
        temp_embedding: Optional[npt.ArrayLike] = None,
    ) -> npt.NDArray[Any]:
        """
        Project `data` into the low-dimensional template space using the chosen
        transform method.

        Parameters
        ----------
        data : array-like, shape (n_samples, n_features) or DataStruct
            High-dimensional data to embed.
        transform_method : str, optional
            Transform method to use. If None, uses the instance `transform_method`.
            Supported: "umap", "knn", "sklearn_rf".
        n_trees : int, optional
            Number of trees/estimators for tree-based regressors (if applicable).
        k : int, optional
            k for KNN-based transforms.
        template : array-like, shape (n_template_samples, n_features), optional
            High-dimensional template data. If None, uses `self.template`.
        temp_embedding : array-like, shape (n_template_samples, n_components), optional
            Low-dimensional embedding of `template`. If None, uses `self.temp_embedding`.

        Returns
        -------
        embed_vals : ndarray, shape (n_samples, n_components)
            Low-dimensional embedding for `data`.
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
            if self.embedder is None:
                raise ValueError("embedder is not set for UMAP transform.")
            embed_vals = self.embedder.transform(data)

        elif transform_method == "knn":
            print("Predicting using KNN")
            raise NotImplementedError

        elif transform_method == "sklearn_rf":
            from sklearn.ensemble import RandomForestRegressor

            rf_embed = RandomForestRegressor(n_estimators=n_trees, n_jobs=-1)
            rf_embed = rf_embed.fit(np.asarray(template), np.asarray(temp_embedding))
            embed_vals = rf_embed.predict(np.asarray(data))

        else:
            raise ValueError(f"Unknown transform_method: {transform_method}")

        print("Total Time embedding: ", time.time() - start)
        return np.asarray(embed_vals)


class BatchEmbed(Embed):
    """
    Batch-based template construction.

    This class repeatedly embeds batches of data, computes dense regions using
    a watershed on a Gaussian density map, and samples representative points from
    each discovered cluster to assemble a global template for a final embedding.

    Parameters
    ----------
    sampling_n : int
        Number of points to sample per cluster for the global template.
    sigma : int
        Gaussian smoothing sigma used when computing density maps.
    batch_method : str
        Embedding method used to embed each batch (e.g. "fitsne", "umap").
    Other parameters
        Inherited from [`Embed`](neuroposelib.embed.Embed).
    """

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
        embedder: Any = None,
        template: Optional[npt.ArrayLike] = None,
        temp_idx: Optional[List[int]] = None,
        temp_embedding: Optional[npt.ArrayLike] = None,
    ) -> None:
        """
        Notes
        -----
        t-SNE parameters here are used in the embedding of batches,
        not for the final template itself.
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
        self.temp_idx = [] if temp_idx is None else temp_idx

    def fit(
        self,
        data: Union[npt.ArrayLike, ds.DataStruct],
        batch_id: Optional[Union[npt.ArrayLike, List[Union[int, str]]]] = None,
        embed_temp: bool = True,
    ) -> "BatchEmbed":
        """
        Build a template by embedding each batch and sampling cluster representatives.

        Parameters
        ----------
        data : array-like, shape (n_frames, n_features)
            Full dataset to sample batches from.
        batch_id : array-like, shape (n_frames,), optional
            Array indicating batch membership for each row in `data`.
        embed_temp : bool, default True
            If True, embed the assembled template into low-dimensional space and save it.

        Returns
        -------
        self : BatchEmbed
            The fitted BatchEmbed instance (with `self.template` and `self.temp_idx` populated).
        """
        self.template = np.empty((0, np.asarray(data).shape[1]))
        self.temp_idx = []
        if batch_id is None:
            unique_batches = np.unique(np.zeros(np.asarray(data).shape[0], dtype=int))
        else:
            unique_batches = np.unique(batch_id)

        for batch in tqdm.tqdm(unique_batches):
            data_by_ID = np.asarray(data)[np.asarray(batch_id) == batch, :]  # Subsetting data by batch

            embed_vals = self.embed(
                data_by_ID, method=self.batch_method, save_self=False
            )

            ws = Watershed(
                sigma=self.sigma, n_bins=1000, max_clip=1, log_out=True, pad_factor=0
            )
            cluster_labels = ws.fit_predict(embed_vals)

            sampled_points, idx = self.__sample_clusters(
                data_by_ID, cluster_labels, sample_size=self.sampling_n
            )

            if batch_id is None:
                # fallback indexing when batch_id is None
                idx = np.array(idx)
            else:
                idx = np.nonzero(np.asarray(batch_id) == batch)[0][idx]

            self.template = np.append(self.template, sampled_points, axis=0)
            self.temp_idx += list(idx)

        if embed_temp:
            self.embed(self.template, save_self=True)  # template would get saved twice

        return self

    def fit_predict(
        self,
        data: Union[npt.ArrayLike, ds.DataStruct],
        batch_id: Optional[Union[npt.ArrayLike, List[Union[int, str]]]] = None,
        save_batchmaps: Optional[str] = None,
    ) -> npt.NDArray[Any]:
        """
        Convenience method: run `fit` then predict embeddings for `data`.

        Parameters
        ----------
        data : array-like, shape (n_frames, n_features)
            Full dataset to embed.
        batch_id : array-like, shape (n_frames,), optional
            Batch membership array.
        save_batchmaps : str, optional
            Path prefix to save batch maps (not currently used).

        Returns
        -------
        embed_vals : ndarray, shape (n_samples, n_components)
            Low-dimensional embedding for `data`.
        """
        self.fit(
            data=data, batch_id=batch_id, embed_temp=True
        )
        embed_vals = self.predict(data)

        return np.asarray(embed_vals)

    def __sample_clusters(
        self,
        data: npt.ArrayLike,
        meta_name: Union[npt.ArrayLike, List[Union[int, str]]],
        sample_size: int = 20,
    ) -> Tuple[npt.NDArray[Any], List[int]]:
        """
        Equally sample points from each cluster.

        Parameters
        ----------
        data : array-like, shape (n_points, n_features)
            Data to sample from.
        meta_name : array-like, shape (n_points,)
            Cluster label for each point in `data`.
        sample_size : int, default 20
            Number of points to sample from each cluster.

        Returns
        -------
        sampled_points : ndarray, shape (m, n_features)
            Array of sampled points where m is the total number of sampled points.
        idx : list of int
            Indices (in `data`) for the sampled points.
        """
        data = np.append(
            np.asarray(data), np.expand_dims(np.arange(np.shape(data)[0]), axis=1), axis=1
        )
        sampled_points = np.empty((0, np.shape(data)[1]))
        for meta_id in np.unique(meta_name):
            points = data[np.asarray(meta_name) == meta_id, :]
            if len(points) < sample_size:
                # If fewer points, just skip (probably artifactual cluster)
                continue
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

    def save_pickle(self, filepath: str = "./plot_folder/") -> "BatchEmbed":
        """
        Save this BatchEmbed instance to a pickle file.

        Parameters
        ----------
        filepath : str, default "./plot_folder/"
            Directory or prefix to write the pickle file; file will be saved as
            `<filepath>batch_embed.p`.

        Returns
        -------
        self : BatchEmbed
            The same instance (for chaining).
        """
        pickle.dump(self, open("".join([filepath, "batch_embed.p"]), "wb"))
        return self

    def load_pickle(self, filepath: str = "./plot_folder/batch_embed.p") -> "BatchEmbed":
        """
        Load a BatchEmbed instance from a pickle file.

        Parameters
        ----------
        filepath : str, default "./plot_folder/batch_embed.p"
            Path to pickle file.

        Returns
        -------
        BatchEmbed
            Loaded BatchEmbed instance.
        """
        loaded = pickle.load(open(filepath, "rb"))
        if not isinstance(loaded, BatchEmbed):
            raise TypeError("Loaded pickle is not a BatchEmbed object.")
        
        self = loaded
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
    Class to create Gaussian-smoothed density maps from 2D point clouds.

    Parameters
    ----------
    sigma : int
        Standard deviation for Gaussian smoothing (in histogram-bin units).
    n_bins : int
        Number of bins along each axis for the 2D histogram.
    max_clip : float
        Fraction of the maximum density to clip values to for visualization.
    log_out : bool
        If True, apply log1p to the smoothed density output.
    pad_factor : float
        Fractional padding applied to histogram range.
    """

    def __init__(
        self,
        sigma: int = 15,
        n_bins: int = 1000,
        max_clip: float = 1,
        log_out: bool = False,
        pad_factor: float = 0.025,
    ) -> None:
        self.sigma = sigma
        self.n_bins = n_bins
        self.max_clip = max_clip
        self.log_out = log_out
        self.pad_factor = pad_factor

        # [[xmin, xmax], [ymin, ymax]]
        self.hist_range: Optional[List[List[float]]] = None

        # TODO: More consideration for when these save
        self.density: Optional[npt.NDArray[np.float64]] = None
        self.data_in_bin: Optional[npt.NDArray[np.int_]] = None
        self.xedges: Optional[npt.NDArray[Any]] = None
        self.yedges: Optional[npt.NDArray[Any]] = None

    def hist(self, data: npt.ArrayLike, new: bool = True) -> npt.NDArray[np.float64]:
        """
        Run a 2D histogram on `data`.

        Parameters
        ----------
        data : array-like, shape (n_points, 2)
            XY coordinates to histogram.
        new : bool, default True
            If False, reuse existing histogram range and bins.

        Returns
        -------
        hist : ndarray, shape (n_bins, n_bins)
            2D histogram (rotated) of the input points.
        """
        data = np.asarray(data)
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

        return hist.astype(np.float64)

    def fit_density(self, data: npt.ArrayLike, new: bool = True, map_bin: bool = True) -> npt.NDArray[np.float64]:
        """
        Calculate Gaussian density for 2D embedding.

        Parameters
        ----------
        data : array-like, shape (n_points, 2)
            2D points.
        new : bool, default True
            If True, compute histogram ranges anew.
        map_bin : bool, default True
            If True, compute and store `self.data_in_bin` mapping points to bins.

        Returns
        -------
        density : ndarray, shape (n_bins, n_bins)
            Smoothed density map.
        """
        data = np.asarray(data)
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
            self.density = density.astype(np.float64)

        return density.astype(np.float64)

    def map_bins(self, data: npt.ArrayLike) -> npt.NDArray[np.int_]:
        """
        Map each 2D point to the corresponding histogram bin indices.

        Parameters
        ----------
        data : array-like, shape (n_points, 2)
            2D points to map.

        Returns
        -------
        data_in_bin : ndarray of int, shape (n_points, 2)
            Array of (row, col) bin indices into density/histogram arrays.
        """
        data = np.asarray(data)

        if getattr(self, "xedges", None) is None:
            print("Could not find histogram, computing now")
            self.density = None
            self.hist(data, new=True)

        dtype = np.int32 if data.dtype == np.float32 else int

        data_in_bin = np.zeros(np.shape(data), dtype)

        for i in range(data_in_bin.shape[0]):
            data_in_bin[i, 1] = (
                np.argmax(self.xedges > data[i, 0]) - 1
            )
            data_in_bin[i, 0] = (
                self.n_bins - np.argmax(self.yedges > data[i, 1]) - 1
            )

        return data_in_bin.astype(np.int_)

    # def plot_density(self, filepath: str = "./plot_folder/density.png"):
    #     f = plt.figure()
    #     ax = f.add_subplot(111)
    #     ax.imshow(self.density)
    #     ax.set_aspect("auto")
    #     plt.savefig(filepath, dpi=400)
    #     plt.close()


class Watershed(GaussDensity):
    """
    Watershed-based clustering on Gaussian density maps.

    Parameters
    ----------
    sigma : int
        Gaussian smoothing sigma (in histogram-bin units).
    n_bins : int
        Number of histogram bins per axis.
    max_clip : float
        Clip fraction for density values.
    log_out : bool
        Whether to apply log1p to the smoothed density.
    pad_factor : float
        Padding fraction when computing histogram ranges.
    density_thresh : float
        Threshold used to define mask for watershed (before internal normalization).
    """

    def __init__(
        self,
        sigma: int = 15,
        n_bins: int = 1000,
        max_clip: float = 0.75,
        log_out: bool = False,
        pad_factor: float = 0.025,
        density_thresh: float = 17,
    ) -> None:
        super().__init__(
            sigma=sigma,
            n_bins=n_bins,
            max_clip=max_clip,
            log_out=log_out,
            pad_factor=pad_factor,
        )
        self.density_thresh = density_thresh / (n_bins**2)

        self.watershed_map: Optional[npt.NDArray[Any]] = None
        self.borders: Optional[dict] = None

        self.density = None  # TODO: Consider more when this saves and doesn't

    def fit(self, data: npt.ArrayLike, sav_threshold: Optional[float] = 0) -> "Watershed":
        """
        Run watershed segmentation on the smoothed density map computed from `data`.

        Parameters
        ----------
        data : array-like, shape (n_points, 2)
            2D points to compute watershed over.
        sav_threshold : float, optional
            If >0, run `merge_clusters` step using this threshold.

        Returns
        -------
        self : Watershed
            The fitted Watershed instance with `watershed_map` and `borders` set.
        """
        from skimage.segmentation import watershed

        self.density = self.fit_density(data, new=True, map_bin=False)

        print("Calculating watershed")
        self.watershed_map = watershed(
            -self.density, mask=self.density > self.density_thresh, watershed_line=False
        )
        self.watershed_map[self.density < self.density_thresh] = 0
        self.borders = {}
        for i in range(1, int(self.watershed_map.max()) + 1):
            # measure.find_contours returns a list; we take first contour found
            contours = measure.find_contours(self.watershed_map.T == i, 0.5)
            self.borders[i] = contours[0] if len(contours) > 0 else np.empty((0, 2))

        if sav_threshold > 0:
            self.merge_clusters(sav_threshold=sav_threshold)

        return self

    def merge_clusters(self, sav_threshold: float) -> "Watershed":
        """
        Merge thin / skinny clusters based on SAV (size / border-length) metric.

        Parameters
        ----------
        sav_threshold : float
            Threshold under which clusters are considered 'thin' and will be merged.

        Returns
        -------
        self : Watershed
            Updated Watershed instance with merged clusters.
        """
        print("Merging thin clusters ...")
        original_borders = self.borders.copy()
        n_clusters = int(self.watershed_map.max()) + 1
        counter = 0
        sav = np.array(
            [
                int(np.sum(self.watershed_map == i)) / len(original_borders[i])
                for i in range(1, n_clusters)
            ]
        )
        slim_clusters = np.where((sav < sav_threshold) & (sav > 0))[0] + 1
        while (len(slim_clusters) > 0) and (counter < 100):
            smallest_cluster = slim_clusters[np.argmin(sav[slim_clusters - 1])]
            border_set = set(map(tuple, self.borders[smallest_cluster]))
            len_overlaps = [
                (
                    len(border_set & set(map(tuple, self.borders[i])))
                    if i in self.borders.keys()
                    else 0
                )
                for i in range(1, n_clusters)
            ]

            len_overlaps[smallest_cluster - 1] = 0
            self.watershed_map = np.where(
                self.watershed_map == smallest_cluster,
                np.argmax(len_overlaps) + 1,
                self.watershed_map,
            )
            self.borders = {}
            for i in range(1, n_clusters):
                bool_map = self.watershed_map.T == i
                if bool_map.sum() > 0:
                    contours = measure.find_contours(self.watershed_map.T == i, 0.5)
                    self.borders[i] = contours[0] if len(contours) > 0 else np.empty((0, 2))

            sav = np.array(
                [
                    int(np.sum(self.watershed_map == i)) / len(original_borders[i])
                    for i in range(1, n_clusters)
                ]
            )
            slim_clusters = np.where((sav < sav_threshold) & (sav > 0))[0] + 1
            counter += 1

        # Fixing cluster labels so that there are no skipped values
        self.borders = dict(sorted(self.borders.items()))
        border_keys = list(self.borders.keys())
        for i, k in enumerate(border_keys):
            self.borders[i + 1] = self.borders.pop(k)
            self.watershed_map[self.watershed_map == k] = i + 1
        return self

    def predict(self, data: Union[ds.DataStruct, npt.ArrayLike]) -> npt.NDArray[np.int_]:
        """
        Predict cluster labels for `data` using the previously computed watershed map.

        Parameters
        ----------
        data : array-like, shape (n_points, 2)
            2D coordinates to label.

        Returns
        -------
        cluster_labels : ndarray of int, shape (n_points,)
            Integer cluster label for each input point.
        """
        data = np.asarray(data)
        dtype = np.int32 if data.dtype == np.float32 else int
        data_in_bin = self.map_bins(data)

        cluster_labels = self.watershed_map[
            data_in_bin[:, 0].astype(dtype), data_in_bin[:, 1].astype(dtype)
        ]
        print(str(int(np.amax(cluster_labels) + 1)), "clusters detected")
        print(str(np.unique(cluster_labels).shape), "unique clusters detected")
        print(np.unique(cluster_labels))

        return np.asarray(cluster_labels).astype(np.int_)

    def fit_predict(
        self,
        data: Optional[Union[ds.DataStruct, npt.ArrayLike]] = None,
        sav_threshold: Optional[float] = 0,
    ) -> npt.NDArray[np.int_]:
        """
        Fit watershed on `data` then predict cluster labels for the same `data`.

        Parameters
        ----------
        data : array-like, shape (n_points, 2)
            2D points to cluster.
        sav_threshold : float, optional
            SAV threshold used in merging step.

        Returns
        -------
        cluster_labels : ndarray of int, shape (n_points,)
            Integer cluster labels for each input point.
        """
        self.fit(data, sav_threshold=sav_threshold)
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
