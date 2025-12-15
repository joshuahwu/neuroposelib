## Unsupervised Behavioral Phenotyping with 3D Pose

Neurodegenerative diseases (like Parkinson's) are characterized by a wide variety of behavioral defects or movement deficits. However, behavior and movement have historically been difficult to quantify and measure. Recent developments in hardware and machine learning have enabled more objective behavioral metrics by providing continuous 3D measurements of naturalistic animal behavior through multi-view videos. These new modalities of data offer a means by which we can comprehensively characterize behavioral phenotypes of neural (dys)-function. We present `neuroposelib` to establish an open-source API with easy access to machine learning methods for the analysis of 3D pose sequences.

This notebook implements a Python version of [CAPTURE (Marshall, 2020)](https://www.cell.com/neuron/fulltext/S0896-6273(20)30894-1?_returnURL=https%3A%2F%2Flinkinghub.elsevier.com%2Fretrieve%2Fpii%2FS0896627320308941%3Fshowall%3Dtrue), which was based on earlier work [MotionMapper (Berman, 2014)](https://royalsocietypublishing.org/doi/full/10.1098/rsif.2014.0672) for the analysis of behavioral data.

To follow this notebook, please download the contents of the [demo dataset](https://duke.box.com/v/demo-mouse-poses) into the `/neuroposelib/tutorials/demo_mouse/` directory.

First, we import the necessary modules.


```python
from neuroposelib import read, write
from neuroposelib import vis
import numpy as np
import time
from IPython.display import Video
from pathlib import Path
import matplotlib.pyplot as plt
%matplotlib inline
```

Load pose predictions, keypoint connectivity information, and metadata.


```python
analysis_key = "tutorial"
config = read.config("../configs/" + analysis_key + ".yaml")

pose, ids = read.pose_h5(config["data_path"] + "demo_mouse.h5")

connectivity = read.connectivity_config(
    path=config["skeleton_path"]
)

meta, meta_by_frame = read.meta(config["data_path"] + "demo_meta.csv", ids=ids)

Path(config["out_path"]).mkdir(parents=True, exist_ok=True)
```

`pose` shape (# frames x # keypoints x 3 coordinates).


```python
print("Pose shape (# frames x # keypoints x 3 coordinates): ")
print(pose.shape)
```

    Pose shape (# frames x # keypoints x 3 coordinates): 
    (648000, 18, 3)


`meta` contains categorical information on recording sessions in `pose`. Here, we have loaded in two sessions. Each frame of the `pose` has a session id label in `ids`.


```python
print(meta)
print("\n" + str(ids))
```

       ids AnimalID     Sex       Strain Condition                           Path
    0    0       A0    Male  Adora2a-Cre  Baseline  ./demo_mouse/demo_mouse_0.mat
    1    1       A1  Female  Adora2a-Cre  Baseline  ./demo_mouse/demo_mouse_1.mat
    
    [0 0 0 ... 1 1 1]


`connectivity` contains key information indicating keypoint labels, connectivity, etc.


```python
print("keypoint labels")
print(connectivity.joint_names)
print("\n Keypoint connections")
print(connectivity.links)
```

    keypoint labels
    ['Snout', 'EarR', 'EarL', 'SpineF', 'SpineM', 'Tail_base', 'ForepawR', 'WristR', 'ElbowR', 'ForepawL', 'WristL', 'ElbowL', 'HindpawR', 'AnkleR', 'KneeR', 'HindpawL', 'AnkleL', 'KneeL']
    
     Keypoint connections
    [[ 0  1]
     [ 1  3]
     [ 0  2]
     [ 2  3]
     [ 2  1]
     [ 0  3]
     [ 4  3]
     [ 5  4]
     [ 6  7]
     [ 7  8]
     [ 8  3]
     [ 9 10]
     [10 11]
     [11  3]
     [12 13]
     [13 14]
     [14  5]
     [15 16]
     [16 17]
     [17  5]]


To see more details on how to set up these files. See the tutorial in `set_up.ipynb`.

Let's plot 150 frames from each session.


```python
vis.pose.arena3D(
    pose,
    connectivity,
    frames=[1000, 500000],
    N_FRAMES=150,
    dpi=100,
    VID_NAME="raw.mp4",
    SAVE_ROOT=config["out_path"],
)

Video(config["out_path"] + "vis_aligned.mp4", width=600, height=600)
```

    100%|██████████| 150/150 [00:13<00:00, 11.10it/s]





<video src="./tutorial_files/vis_aligned.mp4" controls  width="600"  height="600">
      Your browser does not support the <code>video</code> element.
    </video>



Skeletons across sessions may not be aligned worldviews. The following code will estimate the floor plane for each session, and rotate to the x-y plane.


```python
from neuroposelib import preprocess

pose_aligned = preprocess.align_floor_by_id(pose=pose, ids=ids, foot_id=12, head_id=0)

vis.pose.arena3D(
    pose_aligned,
    connectivity,
    frames=[1000, 500000],
    N_FRAMES=150,
    dpi=100,
    VID_NAME="aligned.mp4",
    SAVE_ROOT=config["out_path"],
)

Video(config["out_path"] + "vis_aligned.mp4", width=600, height=600)
```

      0%|          | 0/2 [00:00<?, ?it/s]

    Fitting and rotating the floor for each video to alignment ...


     50%|█████     | 1/2 [00:00<00:00,  4.12it/s]

    Fitting and rotating the floor for each video to alignment ...


    100%|██████████| 2/2 [00:00<00:00,  4.30it/s]
    100%|██████████| 150/150 [00:12<00:00, 12.48it/s]





<video src="./tutorial_files/vis_aligned.mp4" controls  width="600"  height="600">
      Your browser does not support the <code>video</code> element.
    </video>



You can use the following code to save the new aligned poses for easy access later.


```python
from neuroposelib import write

write.pose_h5(pose_aligned, ids, config["data_path"] + "pose_aligned.h5")
```

For this analysis, we would like to prevent divergence of behavioral representations due to global position. Thus, we will generate an egocentric representation of pose for downstream feature calculation. 

Here, we center the mid-spine to $(0,0,0)$, and rotate the front-spine to the $x+$ direction.


```python
# Provide the mid-spine and the mid-spine -> front-spine indices.
pose = preprocess.rotate_spine(preprocess.center_spine(pose_aligned, keypt_idx=4), vector=(4, 3))

vis.pose.arena3D(
    pose,
    connectivity,
    frames=[50000],
    N_FRAMES=150,
    dpi=100,
    VID_NAME="centered.mp4",
    SAVE_ROOT=config["out_path"],
)

Video(config["out_path"] + "vis_centered.mp4", width=600, height=600)
```

    Centering poses to mid spine ...
    Rotating spine to xz plane ...


    100%|██████████| 150/150 [00:08<00:00, 16.95it/s]





<video src="./tutorial_files/vis_centered.mp4" controls  width="600"  height="600">
      Your browser does not support the <code>video</code> element.
    </video>



In this package, we provide functionality for easily calculating features of interest. 

We will just rearrange egocentric x, y, z coordinates of each keypoint into its own set of features. This code does not calculate anything - it just reshapes the pose and generates labels for each feature.


```python
from neuroposelib import features
# Reshape pose to get egocentric pose features
ego_pose, labels = features.get_ego_pose(pose, connectivity.joint_names)
```

    Reformatting pose to egocentric pose features ... 


Write features to or read features from `.h5` file.


```python
# Write
write.features_h5(ego_pose, labels, path=config["out_path"] + "postural_feats.h5")

# Read
ego_pose, labels = read.features_h5(path=config["out_path"] + "postural_feats.h5")
```

    Features loaded at path ./tutorial_files/postural_feats.h5


It's now time for principal component analysis (PCA). PCA is a dimensionality reduction technique which generates orthogonal axes of high variance upon which to project our data. There are many implementations of PCA, but we will use Facebook's Fast Randomized PCA package (`fbpca`), which is significantly faster than most other implementations.


```python
t = time.time()
pc_feats, pc_labels = features.pca(
    ego_pose, labels, categories=["ego_euc"], n_pcs=5, method="fbpca", random_seed=0
)
print(pc_feats[0,:])
print("PCA time: " + str(time.time() - t))
```

    Calculating principal components ... 


    100%|██████████| 1/1 [00:00<00:00,  1.16it/s]

    [ 5.516014   1.0620689  3.9123766 -1.7435495 -2.459202 ]
    PCA time: 1.2266192436218262


    


Although velocities are calculated over rolling windows, the featurization we have so far still lacks the ability to capture complex temporal signals.

To address this, we can leverage the frequency domain through a Morlet wavelet transformation.

Let's see first what a Morlet wavelet looks like.


```python
import pywt

psi, t = pywt.ContinuousWavelet("cmor1.0-1.0").wavefun(10)
fig = plt.plot(figsize=(12, 10))
plt.plot(t, psi.real, t, psi.imag)
plt.xlabel("Time (s)")
plt.show()
```


    
![png](tutorial_files/tutorial_31_0.png)
    



```python
wlet_feats, wlet_labels = features.wavelet(
    pc_feats, pc_labels, ids, fs=90, freq=np.linspace(1, 25, 25), bw=1.0
)
```

    Calculating wavelets ... 
    Calculating wavelets for video 0
    Calculating wavelets for video 1


We now use PCA to reduce the dimensions of the new wavelet features, and consolidate with the previous PC scores. Each frame is now associated with a vector of features corresponding to the PC scores of egocentric keypoint coordinates and local frequency information.


```python
# PCA on wavelet features
pc_wlet, pc_wlet_labels = features.pca(
    wlet_feats,
    wlet_labels,
    categories=["wlet_ego_euc"],
    n_pcs=5,
    method="fbpca",
    random_seed=0,
)

pc_feats = np.hstack((pc_feats, pc_wlet))
pc_labels += pc_wlet_labels
```

    Calculating principal components ... 


    100%|██████████| 1/1 [00:01<00:00,  1.92s/it]



```python
# Optionally save full PC features to file
write.features_h5(
    pc_feats, pc_labels, path="".join([config["out_path"], "pca_feats.h5"])
)
```

We encapsulate all relevant data to store in a data object.


```python
from neuroposelib import DataStruct as ds

data_obj = ds.DataStruct(
    pose=pose,
    ids=ids,
    meta=meta,
    meta_by_frame=meta_by_frame,
    connectivity=connectivity,
    features = pc_feats,
)

# When using high framerate data, downsampling may be necessary in order to 
# prevent oversmoothing
data_obj = data_obj[:: config["downsample"], :]
```

Using t-SNE, frames are projected onto a 2D embedding for clustering and visualization.


```python
from neuroposelib.embed import Embed

embedder = Embed(
    embed_method=config["single_embed"]["method"],
    perplexity=config["single_embed"]["perplexity"],
    lr=config["single_embed"]["lr"],
    random_seed=0,
)
data_obj.embed_vals = embedder.embed(data_obj.features, save_self=True)
```

    Running fitsne via openTSNE
    --------------------------------------------------------------------------------
    TSNE(early_exaggeration=12, exaggeration=1.5, n_jobs=-1,
         negative_gradient_method='fft', neighbors='annoy', perplexity=50,
         random_state=0, verbose=True)
    --------------------------------------------------------------------------------
    ===> Finding 150 nearest neighbors using Annoy approximate search using euclidean distance...
       --> Time elapsed: 7.94 seconds
    ===> Calculating affinity matrix...
       --> Time elapsed: 3.45 seconds
    ===> Calculating PCA-based initialization...
       --> Time elapsed: 0.01 seconds
    ===> Running optimization with exaggeration=12.00, lr=5400.00 for 250 iterations...
    Iteration   50, KL divergence 5.7138, 50 iterations in 1.6675 sec
    Iteration  100, KL divergence 5.8244, 50 iterations in 1.6998 sec
    Iteration  150, KL divergence 5.8251, 50 iterations in 1.7235 sec
    Iteration  200, KL divergence 5.8251, 50 iterations in 1.7065 sec
    Iteration  250, KL divergence 5.8251, 50 iterations in 1.7089 sec
       --> Time elapsed: 8.51 seconds
    ===> Running optimization with exaggeration=1.50, lr=43200.00 for 500 iterations...
    Iteration   50, KL divergence 3.8416, 50 iterations in 1.6657 sec
    Iteration  100, KL divergence 3.6496, 50 iterations in 1.6967 sec
    Iteration  150, KL divergence 3.5577, 50 iterations in 1.8094 sec
    Iteration  200, KL divergence 3.5010, 50 iterations in 1.9653 sec
    Iteration  250, KL divergence 3.4624, 50 iterations in 2.0866 sec
    Iteration  300, KL divergence 3.4340, 50 iterations in 2.1751 sec
    Iteration  350, KL divergence 3.4123, 50 iterations in 2.3354 sec
    Iteration  400, KL divergence 3.3954, 50 iterations in 2.3043 sec
    Iteration  450, KL divergence 3.3814, 50 iterations in 2.4645 sec
    Iteration  500, KL divergence 3.3694, 50 iterations in 2.5370 sec
       --> Time elapsed: 21.04 seconds


The histogram of the 2D embedding is smoothed with a Gaussian, and segmented by the watershed algorithm to determine cluster assignments.


```python
from neuroposelib.embed import Watershed
# Watershed clustering
data_obj.ws = Watershed(
    sigma=config["single_embed"]["sigma"], max_clip=1, log_out=True, pad_factor=0.05
)
data_obj.data["Cluster"] = data_obj.ws.fit_predict(data=data_obj.embed_vals)

# Plot density
vis.plot.density(
    data_obj.ws.density,
    data_obj.ws.borders,
    filepath=config["out_path"] + "/density.png",
    show=True,
)
```

    Calculating new histogram ranges
    Calculating watershed
    106 clusters detected
    (104,) unique clusters detected
    [  1   2   3   4   5   6   7   8   9  10  11  12  13  14  15  16  17  18
      19  20  21  22  23  24  25  26  27  28  29  30  31  32  33  34  35  36
      37  38  39  40  41  42  43  44  45  46  47  48  49  50  51  52  53  54
      55  56  57  58  59  60  61  62  63  64  66  67  68  69  70  71  72  73
      74  75  76  77  78  79  80  81  82  83  84  85  86  87  88  89  90  91
      92  93  94  95  96  97  98  99 100 101 102 103 104 105]



    
![png](tutorial_files/tutorial_41_1.png)
    


Within the embedding, we can visualize the density of each animal separately.


```python
vis.plot.density_cat(
    data=data_obj,
    column="ids",
    watershed=data_obj.ws,
    filepath=config["out_path"] + "/density_id.png",
    show=True,
)
```


    
![png](tutorial_files/tutorial_43_0.png)
    


We can also randomly sample some actions from each cluster. Videos will save in `neuroposelib/tutorials/results/tutorial/skeleton_vids/`


```python
vis.pose.sample_arena3D(
    pose_aligned,
    connectivity,
    labels=data_obj.data["Cluster"],
    n_samples=9,
    centered=True,
    VID_NAME = "cluster",
    N_FRAMES=100,
    fps=90,
    watershed=data_obj.ws,
    embed_vals=data_obj.embed_vals,
    verbose=False,
    filepath=config["out_path"],
)
```

    Detected labels not the same shape as pose...
    Assuming labels downsampled by 10


      0%|          | 0/104 [00:00<?, ?it/s]

    100%|██████████| 104/104 [2:40:12<00:00, 92.43s/it] 

