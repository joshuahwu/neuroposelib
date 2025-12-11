## Unsupervised Behavioral Phenotyping with 3D Skeletal Pose
Joshua Wu

Duke University Biomedical Engineering

[Timothy Dunn Lab](https://www.tdunnlab.org/)

11 June, 2024

Neurodegenerative diseases (like Parkinson's) are characterized by a wide variety of behavioral defects or movement deficits. However, behavior and movement have historically been difficult to quantify and measure. Recent developments in hardware and machine learning have enabled more objective behavioral metrics by providing continuous 3D measurements of naturalistic animal behavior through multi-view videos. These new modalities of data offer a means by which we can comprehensively characterize behavioral phenotypes of neural (dys)-function. We present `neuroposelib` to establish an open-source API with easy access to machine learning methods for the analysis of 3D pose sequences.

This notebook implements a Python version of [CAPTURE (Marshall, 2020)](https://www.cell.com/neuron/fulltext/S0896-6273(20)30894-1?_returnURL=https%3A%2F%2Flinkinghub.elsevier.com%2Fretrieve%2Fpii%2FS0896627320308941%3Fshowall%3Dtrue), which was based on earlier work [MotionMapper (Berman, 2014)](https://royalsocietypublishing.org/doi/full/10.1098/rsif.2014.0672) for the analysis of behavioral data.

To follow this notebook, please download the contents of the [demo dataset](https://duke.box.com/v/demo-mouse-poses) into the `/neuroposelib/tutorials/demo_mouse/` directory.

First, we import the necessary modules.


```python
from neuroposelib import read
from neuroposelib import vis
import numpy as np
import time
from IPython.display import Video
from pathlib import Path
import matplotlib.pyplot as plt
%matplotlib inline
```

    /mnt/sw/nix/store/71ksmx7k6xy3v9ksfkv5mp5kxxp64pd6-python-3.10.13-view/lib/python3.10/site-packages/numpy/core/getlimits.py:549: UserWarning: The value of the smallest subnormal for <class 'numpy.float64'> type is zero.
      setattr(self, word, getattr(machar, word).flat[0])
    /mnt/sw/nix/store/71ksmx7k6xy3v9ksfkv5mp5kxxp64pd6-python-3.10.13-view/lib/python3.10/site-packages/numpy/core/getlimits.py:89: UserWarning: The value of the smallest subnormal for <class 'numpy.float64'> type is zero.
      return self._float_to_str(self.smallest_subnormal)
    /mnt/sw/nix/store/71ksmx7k6xy3v9ksfkv5mp5kxxp64pd6-python-3.10.13-view/lib/python3.10/site-packages/numpy/core/getlimits.py:549: UserWarning: The value of the smallest subnormal for <class 'numpy.float32'> type is zero.
      setattr(self, word, getattr(machar, word).flat[0])
    /mnt/sw/nix/store/71ksmx7k6xy3v9ksfkv5mp5kxxp64pd6-python-3.10.13-view/lib/python3.10/site-packages/numpy/core/getlimits.py:89: UserWarning: The value of the smallest subnormal for <class 'numpy.float32'> type is zero.
      return self._float_to_str(self.smallest_subnormal)


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

Video(config["out_path"] + "vis_raw.mp4", width=600, height=600)
```

      0%|          | 0/150 [00:00<?, ?it/s]

    100%|██████████| 150/150 [00:16<00:00,  9.31it/s]





<!-- <video src="./results/tutorial/vis_raw.mp4" controls  width="600"  height="600">
      Your browser does not support the <code>video</code> element.
    </video> -->



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


     50%|█████     | 1/2 [00:00<00:00,  4.34it/s]

    Fitting and rotating the floor for each video to alignment ... 


    100%|██████████| 2/2 [00:00<00:00,  4.37it/s]
    100%|██████████| 150/150 [00:12<00:00, 12.31it/s]





<!-- <video src="./results/tutorial/vis_aligned.mp4" controls  width="600"  height="600">
      Your browser does not support the <code>video</code> element.
    </video> -->



You can use the following code to save the new aligned poses for easy access later.


```python
from neuroposelib import write

# write.pose_h5(pose_aligned, ids, config["data_path"] + "pose_aligned.h5")
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


    100%|██████████| 150/150 [00:08<00:00, 17.91it/s]





<!-- <video src="./results/tutorial/vis_centered.mp4" controls  width="600"  height="600">
      Your browser does not support the <code>video</code> element.
    </video> -->



In this package, we provide functionality for easily calculating features of interest. 

Using this centered and spine-locked pose transformation, we can calculate relative velocities of all keypoints. We leave out the mid spine since it is centered.


```python
from neuroposelib import features

# # Getting relative velocities
# rel_vel, rel_vel_labels = features.get_velocities(
#     pose,
#     ids,
#     connectivity.joint_names,
#     joints=np.delete(np.arange(18), 4),
#     widths=[5, 11, 51],
# )
```

You can also calculate joint angles of interest as specified in `skeletons.py`.


```python
print(connectivity.angles)
```

    [[ 0  1  3]
     [ 0  2  3]
     [ 0  3  4]
     [ 1  3  4]
     [ 2  3  4]
     [ 3  4  5]
     [ 1  3  8]
     [ 2  3  8]
     [ 0  3  8]
     [ 3  8  7]
     [ 8  7  6]
     [ 1  3 11]
     [ 2  3 11]
     [ 0  3 11]
     [ 3 11 10]
     [11 10  9]
     [ 4  5 14]
     [ 5 14 13]
     [14 13 12]
     [ 4  5 17]
     [ 5 17 16]
     [17 16 15]
     [ 0  3  6]
     [ 0  3  7]
     [ 0  3  9]
     [ 0  3 10]
     [ 4  5 12]
     [ 4  5 13]
     [ 4  5 15]
     [ 4  5 16]]



```python
# Calculating joint angles
angles, angle_labels = features.get_angles(pose, connectivity.angles)
```

    Calculating joint angles ... 


      0%|          | 0/30 [00:00<?, ?it/s]

    100%|██████████| 30/30 [00:00<00:00, 35.02it/s]


These velocity and angle calculations are just for demonstration, we will not use velocities or angles for the analysis in this tutorial.

We will just rearrange egocentric x, y, z coordinates of each keypoint into its own set of features. This code does not calculate anything - it just reshapes the pose and generates labels for each feature.


```python
# Reshape pose to get egocentric pose features
ego_pose, labels = features.get_ego_pose(pose, connectivity.joint_names)

# Clear some memory
del angles, angle_labels, #rel_vel, rel_vel_labels
```

    Reformatting pose to egocentric pose features ... 


Write features to or read features from `.h5` file.


```python
# Write
# write.features_h5(features, labels, path=config["out_path"] + "postural_feats.h5")

# Read
# features, labels = read.features_h5(path=config["out_path"] + "postural_feats.h5")
```

It's now time for principal component analysis (PCA). PCA is a dimensionality reduction technique which generates orthogonal axes of high variance upon which to project our data. There are many implementations of PCA, but we will use Facebook's Fast Randomized PCA package (`fbpca`), which is significantly faster than most other implementations.


```python
t = time.time()
pc_feats, pc_labels = features.pca(
    ego_pose, labels, categories=["ego_euc"], n_pcs=5, method="fbpca"
)
print("PCA time: " + str(time.time() - t))

del ego_pose, labels
```

    Calculating principal components ... 


    100%|██████████| 1/1 [00:00<00:00,  1.13it/s]

    PCA time: 1.340679407119751


    


Although velocities are calculated over rolling windows, the featurization we have so far still lacks the ability to capture complex temporal signals.

To address this, we can leverage the frequency domain through a Morlet wavelet transformation.

Let's see first what a Morlet wavelet looks like.


```python
from scipy import signal
M = 100
w0 = 5
s = w0*90/(2*np.pi*25)
morlet_wavelet = signal.morlet2(M, s, w0)
plt.plot(morlet_wavelet.imag, label='Imaginary')
plt.plot(morlet_wavelet.real, label='Real')
plt.legend()
plt.show()
```

    /tmp/ipykernel_1364393/230954268.py:5: DeprecationWarning: scipy.signal.morlet2 is deprecated in SciPy 1.12 and will be removed
    in SciPy 1.15. We recommend using PyWavelets instead.
    
      morlet_wavelet = signal.morlet2(M, s, w0)



    
![png](./tutorial_files/tutorial_35_1.png)
    



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
)

del wlet_feats, wlet_labels
pc_feats = np.hstack((pc_feats, pc_wlet))
pc_labels += pc_wlet_labels
del pc_wlet, pc_wlet_labels
```

    Calculating principal components ... 


    100%|██████████| 1/1 [00:01<00:00,  1.99s/it]



```python
# Optionally save full PC features to file
# write.features_h5(
#     pc_feats, pc_labels, path="".join([config["out_path"], "pca_feats.h5"])
# )
```

We encapsulate all relevant data to store in a data object.


```python
from neuroposelib import DataStruct as ds

data_obj = ds.DataStruct(
    pose=pose,
    id=ids,
    meta=meta,
    meta_by_frame=meta_by_frame,
    connectivity=connectivity,
)

data_obj.features = pc_feats
# When using high framerate data, downsampling may be necessary in order to 
# discover granular structure in the embedding
data_obj = data_obj[:: config["downsample"], :]
```

Using t-SNE, frames are projected onto a 2D embedding for clustering and visualization.


```python
from neuroposelib.embed import Embed

embedder = Embed(
    embed_method=config["single_embed"]["method"],
    perplexity=config["single_embed"]["perplexity"],
    lr=config["single_embed"]["lr"],
)
data_obj.embed_vals = embedder.embed(data_obj.features, save_self=True)
```

    Running fitsne via openTSNE
    --------------------------------------------------------------------------------
    TSNE(early_exaggeration=12, exaggeration=1.5, n_jobs=-1,
         negative_gradient_method='fft', neighbors='annoy', perplexity=50,
         verbose=True)
    --------------------------------------------------------------------------------
    ===> Finding 150 nearest neighbors using Annoy approximate search using euclidean distance...
       --> Time elapsed: 9.02 seconds
    ===> Calculating affinity matrix...
       --> Time elapsed: 3.50 seconds
    ===> Calculating PCA-based initialization...
       --> Time elapsed: 0.06 seconds
    ===> Running optimization with exaggeration=12.00, lr=5400.00 for 250 iterations...
    Iteration   50, KL divergence 5.7024, 50 iterations in 1.7998 sec
    Iteration  100, KL divergence 5.8145, 50 iterations in 1.8229 sec
    Iteration  150, KL divergence 5.8152, 50 iterations in 1.8565 sec
    Iteration  200, KL divergence 5.8152, 50 iterations in 1.7971 sec
    Iteration  250, KL divergence 5.8152, 50 iterations in 1.7784 sec
       --> Time elapsed: 9.06 seconds
    ===> Running optimization with exaggeration=1.50, lr=43200.00 for 500 iterations...
    Iteration   50, KL divergence 3.8435, 50 iterations in 1.7515 sec
    Iteration  100, KL divergence 3.6488, 50 iterations in 1.7909 sec
    Iteration  150, KL divergence 3.5556, 50 iterations in 1.9389 sec
    Iteration  200, KL divergence 3.4996, 50 iterations in 2.0700 sec
    Iteration  250, KL divergence 3.4621, 50 iterations in 2.2122 sec
    Iteration  300, KL divergence 3.4343, 50 iterations in 2.2884 sec
    Iteration  350, KL divergence 3.4134, 50 iterations in 2.4896 sec
    Iteration  400, KL divergence 3.3973, 50 iterations in 2.4762 sec
    Iteration  450, KL divergence 3.3840, 50 iterations in 2.5444 sec
    Iteration  500, KL divergence 3.3719, 50 iterations in 2.6212 sec
       --> Time elapsed: 22.18 seconds


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
    Calculating log density
    0.00014402738335640792
    0.00014400544855744678
    Calculating watershed
    107 clusters detected
    (104,) unique clusters detected
    [  1   2   3   4   5   6   7   8   9  10  11  12  13  14  15  16  17  18
      19  20  21  22  23  24  25  26  27  28  29  30  31  32  33  34  35  36
      37  38  39  40  41  42  43  44  45  46  47  48  49  50  51  52  53  54
      55  56  57  58  59  60  61  62  63  64  65  66  67  68  69  71  72  73
      74  75  77  78  79  80  81  82  83  84  85  86  87  88  89  90  91  92
      93  94  95  96  97  98  99 100 101 102 103 104 105 106]



    
![./png](tutorial_files/tutorial_45_1.png)
    


Within the embedding, we can visualize the density of each animal separately.


```python
vis.plot.density_cat(
    data=data_obj,
    column="id",
    watershed=data_obj.ws,
    filepath=config["out_path"] + "/density_id.png",
    show=True,
)
```

    Calculating log density
    0.00014402738335640797
    0.0001440036188855628


    Calculating log density
    0.00014402738335640795
    0.0001440034977567786



    
![./png](tutorial_files/tutorial_47_2.png)
    


We can also randomly sample some actions from each cluster. Videos will save in `neuroposelib/tutorials/results/tutorial/skeleton_vids/`


```python
# vis.pose.arena3D_map(
#     pose_aligned,
#     density=vis.pose._mask_density(data_obj.ws.watershed_map, eps=vis.pose.constants.EPS*1.01),
#     connectivity=connectivity,
#     frames=[465370, 466000, 465650, 164360, 223730, 536540],
#     centered=True,
#     VID_NAME="test",
#     N_FRAMES=100,
#     fps=90,
#     watershed=data_obj.ws,
# )
```


    ---------------------------------------------------------------------------

    NameError                                 Traceback (most recent call last)

    Cell In[1], line 1
    ----> 1 vis.pose.arena3D_map(
          2     pose_aligned,
          3     density=vis.pose._mask_density(data_obj.ws.watershed_map, eps=vis.pose.constants.EPS*1.01),
          4     connectivity=connectivity,
          5     frames=[465370, 466000, 465650, 164360, 223730, 536540],
          6     centered=True,
          7     VID_NAME="test",
          8     N_FRAMES=100,
          9     fps=90,
         10     watershed=data_obj.ws,
         11 )


    NameError: name 'vis' is not defined



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
    filepath=config["out_path"],
)
```

    Detected labels not the same shape as pose...
    Assuming labels downsampled by 10


      0%|          | 0/104 [00:00<?, ?it/s]

    [64780, 540120, 384780, 278110, 548280, 406660, 77230, 310420, 83410]
    Calculating log density
    0.0001440273833564081
    0.00014283748281496147


    100%|██████████| 100/100 [01:31<00:00,  1.09it/s]
      1%|          | 1/104 [01:32<2:38:32, 92.35s/it]

    [449950, 193140, 74950, 496090, 341170, 30030, 57710, 235580, 193970]
    Calculating log density
    0.0001440273833564082
    0.000142853672160559


    100%|██████████| 100/100 [01:31<00:00,  1.09it/s]
      2%|▏         | 2/104 [03:04<2:37:09, 92.44s/it]

    [262840, 395370, 521360, 85460, 66790, 86870, 306700, 5430, 271170]
    Calculating log density
    0.0001440273833564083
    0.0001425270934636902


    100%|██████████| 100/100 [01:32<00:00,  1.08it/s]
      3%|▎         | 3/104 [04:38<2:36:19, 92.86s/it]

    [311260, 192930, 406600, 6000, 421410, 248680, 5050, 202030, 398790]
    Calculating log density
    0.000144027383356408
    0.00014280410766386823


    100%|██████████| 100/100 [01:32<00:00,  1.08it/s]
      4%|▍         | 4/104 [06:11<2:34:57, 92.97s/it]

    [18930, 52400, 333140, 522290, 174440, 90820, 400590, 475420, 539350]
    Calculating log density
    0.00014402738335640776
    0.0001426349496110524


    100%|██████████| 100/100 [01:32<00:00,  1.08it/s]
      5%|▍         | 5/104 [07:44<2:33:21, 92.94s/it]

    [550010, 253920, 334300, 268710, 523240, 380940, 459820, 190040, 190590]
    Calculating log density
    0.0001440273833564077
    0.00014271417165809123


    100%|██████████| 100/100 [01:32<00:00,  1.08it/s]
      6%|▌         | 6/104 [09:17<2:32:03, 93.09s/it]

    [524390, 472210, 404340, 393650, 607040, 376420, 274920, 48590, 66560]
    Calculating log density
    0.0001440273833564079
    0.0001411196200169298


    100%|██████████| 100/100 [01:32<00:00,  1.09it/s]
      7%|▋         | 7/104 [10:50<2:30:14, 92.93s/it]

    [524850, 267650, 19030, 64840, 562390, 365760, 309040, 429870, 358190]
    Calculating log density
    0.00014402738335640806
    0.000142009804204066


    100%|██████████| 100/100 [01:33<00:00,  1.07it/s]
      8%|▊         | 8/104 [12:23<2:29:05, 93.19s/it]

    [3140, 37780, 378850, 352720, 577400, 213600, 169350, 54380, 480610]
    Calculating log density
    0.0001440273833564053
    0.00014110088867476866


    100%|██████████| 100/100 [01:33<00:00,  1.07it/s]
      9%|▊         | 9/104 [13:57<2:27:47, 93.34s/it]

    [643890, 146200, 471310, 524500, 560350, 237110, 318120, 122690, 451590]
    Calculating log density
    0.00014402738335640727
    0.000141183383702992


    100%|██████████| 100/100 [01:33<00:00,  1.07it/s]
     10%|▉         | 10/104 [15:31<2:26:38, 93.60s/it]

    [472430, 44950, 292200, 538580, 397540, 302560, 195970, 386080, 53960]
    Calculating log density
    0.00014402738335640738
    0.00014244072633811322


    100%|██████████| 100/100 [01:32<00:00,  1.08it/s]
     11%|█         | 11/104 [17:05<2:25:08, 93.64s/it]

    [336180, 634360, 486990, 550860, 395450, 628320, 629820, 571180, 121600]
    Calculating log density
    0.00014402738335640776
    0.00014176638534500773


    100%|██████████| 100/100 [01:32<00:00,  1.08it/s]
     12%|█▏        | 12/104 [18:38<2:23:20, 93.48s/it]

    [268300, 340700, 256240, 600120, 446950, 254320, 380530, 346580, 106200]
    Calculating log density
    0.00014402738335640806
    0.00014252253215058926


    100%|██████████| 100/100 [01:33<00:00,  1.07it/s]
     12%|█▎        | 13/104 [20:12<2:21:56, 93.59s/it]

    [50950, 77910, 266770, 644830, 326080, 408390, 64920, 117480, 102740]
    Calculating log density
    0.000144027383356408
    0.00014318355660271142


    100%|██████████| 100/100 [01:32<00:00,  1.08it/s]
     13%|█▎        | 14/104 [21:45<2:20:10, 93.45s/it]

    [496570, 18750, 564590, 579740, 48530, 320750, 201650, 90050, 213530]
    Calculating log density
    0.000144027383356408
    0.00014344711222548818


    100%|██████████| 100/100 [01:32<00:00,  1.08it/s]
     14%|█▍        | 15/104 [23:19<2:18:37, 93.45s/it]

    [249670, 568910, 567880, 124390, 91930, 538480, 341620, 569180, 317060]
    Calculating log density
    0.00014402738335640922
    0.00014211825397969076


    100%|██████████| 100/100 [01:32<00:00,  1.08it/s]
     15%|█▌        | 16/104 [24:52<2:16:55, 93.36s/it]

    [305700, 198990, 564160, 63500, 192650, 423260, 539910, 566470, 434810]
    Calculating log density
    0.00014402738335640835
    0.00014217106923220945


    100%|██████████| 100/100 [01:32<00:00,  1.08it/s]
     16%|█▋        | 17/104 [26:25<2:15:25, 93.40s/it]

    [561470, 7600, 482820, 107620, 325870, 451760, 71760, 324660, 350830]
    Calculating log density
    0.0001440273833564083
    0.00014225365034945905


    100%|██████████| 100/100 [01:32<00:00,  1.08it/s]
     17%|█▋        | 18/104 [27:58<2:13:42, 93.29s/it]

    [352590, 38040, 73580, 343320, 431060, 330040, 129230, 378310, 268730]
    Calculating log density
    0.00014402738335640857
    0.00014228130564730213


    100%|██████████| 100/100 [01:32<00:00,  1.08it/s]
     18%|█▊        | 19/104 [29:31<2:11:58, 93.16s/it]

    [62970, 409540, 39530, 255840, 24160, 498020, 24850, 256480, 155810]
    Calculating log density
    0.00014402738335640881
    0.0001428233985953778


    100%|██████████| 100/100 [01:32<00:00,  1.08it/s]
     19%|█▉        | 20/104 [31:04<2:10:25, 93.17s/it]

    [559890, 350170, 486740, 356770, 74740, 566330, 167530, 423610, 273780]
    Calculating log density
    0.0001440273833564075
    0.0001418564226570156


    100%|██████████| 100/100 [01:32<00:00,  1.08it/s]
     20%|██        | 21/104 [32:38<2:09:01, 93.27s/it]

    [401780, 282410, 328770, 258190, 238170, 619580, 389240, 424240, 459240]
    Calculating log density
    0.00014402738335640743
    0.0001422341993485659


    100%|██████████| 100/100 [01:33<00:00,  1.07it/s]
     21%|██        | 22/104 [34:12<2:07:47, 93.51s/it]

    [489820, 387160, 337000, 427800, 643950, 265920, 339790, 360550, 592970]
    Calculating log density
    0.0001440273833564078
    0.00014266310850761282


    100%|██████████| 100/100 [01:32<00:00,  1.08it/s]
     22%|██▏       | 23/104 [35:45<2:06:09, 93.46s/it]

    [570170, 361840, 259500, 313630, 257140, 82390, 325830, 324890, 529470]
    Calculating log density
    0.0001440273833564077
    0.00014343863129336753


    100%|██████████| 100/100 [01:32<00:00,  1.08it/s]
     23%|██▎       | 24/104 [37:18<2:04:31, 93.40s/it]

    [541950, 483270, 258180, 317140, 241670, 213540, 342600, 259070, 127750]
    Calculating log density
    0.000144027383356407
    0.00014240046917634536


    100%|██████████| 100/100 [01:32<00:00,  1.08it/s]
     24%|██▍       | 25/104 [38:52<2:02:56, 93.37s/it]

    [555910, 296280, 414650, 459080, 350510, 293600, 572820, 157680, 183790]
    Calculating log density
    0.0001440273833564078
    0.00014326246723009924


    100%|██████████| 100/100 [01:33<00:00,  1.07it/s]
     25%|██▌       | 26/104 [40:26<2:01:37, 93.56s/it]

    [195760, 69020, 113020, 491450, 429660, 158750, 608420, 89210, 61390]
    Calculating log density
    0.0001440273833564079
    0.0001423196690048804


    100%|██████████| 100/100 [01:32<00:00,  1.08it/s]
     26%|██▌       | 27/104 [41:59<2:00:03, 93.55s/it]

    [451520, 344800, 106010, 332540, 568990, 520500, 155670, 569220, 381710]
    Calculating log density
    0.0001440273833564086
    0.0001424502646918363


    100%|██████████| 100/100 [01:32<00:00,  1.08it/s]
     27%|██▋       | 28/104 [43:33<1:58:28, 93.53s/it]

    [185610, 562950, 538090, 560650, 84490, 350840, 380210, 890, 112050]
    Calculating log density
    0.00014402738335640806
    0.00014272435872691213


    100%|██████████| 100/100 [01:32<00:00,  1.08it/s]
     28%|██▊       | 29/104 [45:07<1:57:01, 93.62s/it]

    [39130, 436150, 105810, 496490, 78990, 518580, 431360, 495100, 548180]
    Calculating log density
    0.000144027383356408
    0.00014347229772076838


    100%|██████████| 100/100 [01:33<00:00,  1.07it/s]
     29%|██▉       | 30/104 [46:40<1:55:31, 93.67s/it]

    [30190, 416380, 317370, 565110, 576310, 104930, 498950, 254380, 209300]
    Calculating log density
    0.00014402738335640884
    0.00014201296648875321


    100%|██████████| 100/100 [01:33<00:00,  1.07it/s]
     30%|██▉       | 31/104 [48:14<1:53:56, 93.65s/it]

    [114020, 113030, 399470, 265470, 498640, 18990, 40280, 321920, 337850]
    Calculating log density
    0.00014402738335640263
    0.00014076245472776413


    100%|██████████| 100/100 [01:32<00:00,  1.08it/s]
     31%|███       | 32/104 [49:48<1:52:20, 93.62s/it]

    [402680, 613730, 518930, 259400, 260530, 49440, 239800, 196880, 365970]
    Calculating log density
    0.0001440273833564069
    0.00014214089871940368


    100%|██████████| 100/100 [01:32<00:00,  1.08it/s]
     32%|███▏      | 33/104 [51:21<1:50:44, 93.58s/it]

    [388230, 367260, 576080, 473440, 50430, 433160, 290520, 481100, 621710]
    Calculating log density
    0.00014402738335640692
    0.00014224012083866939


    100%|██████████| 100/100 [01:33<00:00,  1.07it/s]
     33%|███▎      | 34/104 [52:55<1:49:16, 93.67s/it]

    [130000, 521170, 293730, 60640, 29080, 377260, 567010, 61490, 29510]
    Calculating log density
    0.00014402738335640795
    0.00014211691586940223


    100%|██████████| 100/100 [01:33<00:00,  1.07it/s]
     34%|███▎      | 35/104 [54:29<1:47:52, 93.81s/it]

    [544380, 390300, 323260, 196980, 189000, 268670, 81240, 551100, 236020]
    Calculating log density
    0.00014402738335640846
    0.00014217891405879786


    100%|██████████| 100/100 [01:33<00:00,  1.07it/s]
     35%|███▍      | 36/104 [56:03<1:46:27, 93.94s/it]

    [378440, 22710, 477070, 399480, 480040, 644870, 639840, 591250, 549450]
    Calculating log density
    0.00014402738335640762
    0.00014172157355766277


    100%|██████████| 100/100 [01:33<00:00,  1.07it/s]
     36%|███▌      | 37/104 [57:37<1:44:58, 94.00s/it]

    [93990, 341940, 45660, 282470, 307390, 514820, 273030, 380320, 189080]
    Calculating log density
    0.00014402738335640749
    0.00014222663475142697


    100%|██████████| 100/100 [01:33<00:00,  1.07it/s]
     37%|███▋      | 38/104 [59:11<1:43:19, 93.93s/it]

    [542190, 69900, 489680, 585500, 258440, 556140, 107540, 410960, 419990]
    Calculating log density
    0.00014402738335640822
    0.00014299912232943268


    100%|██████████| 100/100 [01:32<00:00,  1.08it/s]
     38%|███▊      | 39/104 [1:00:45<1:41:37, 93.81s/it]

    [55910, 52860, 42480, 310210, 279210, 615700, 474880, 530280, 280540]
    Calculating log density
    0.0001440273833564077
    0.000142790903882028


    100%|██████████| 100/100 [01:32<00:00,  1.08it/s]
     38%|███▊      | 40/104 [1:02:18<1:39:55, 93.68s/it]

    [215250, 455100, 153370, 126050, 202340, 13230, 454290, 188610, 377680]
    Calculating log density
    0.0001440273833564085
    0.0001428653459169379


    100%|██████████| 100/100 [01:33<00:00,  1.07it/s]
     39%|███▉      | 41/104 [1:03:52<1:38:21, 93.68s/it]

    [589060, 339930, 377250, 120080, 423290, 486460, 158380, 47280, 121080]
    Calculating log density
    0.00014402738335640795
    0.00014275648711007218


    100%|██████████| 100/100 [01:32<00:00,  1.08it/s]
     40%|████      | 42/104 [1:05:25<1:36:45, 93.63s/it]

    [104630, 92220, 363360, 533340, 205560, 58430, 72460, 190270, 302960]
    Calculating log density
    0.00014402738335640803
    0.00014225457413461794


    100%|██████████| 100/100 [01:33<00:00,  1.07it/s]
     41%|████▏     | 43/104 [1:06:59<1:35:12, 93.65s/it]

    [118630, 309730, 250430, 587610, 454970, 471320, 115490, 53510, 636060]
    Calculating log density
    0.00014402738335640862
    0.00014293974255346047


    100%|██████████| 100/100 [01:33<00:00,  1.07it/s]
     42%|████▏     | 44/104 [1:08:33<1:33:45, 93.76s/it]

    [527990, 388950, 32220, 367000, 483440, 569520, 95150, 522800, 412790]
    Calculating log density
    0.00014402738335640806
    0.0001434330069290076


    100%|██████████| 100/100 [01:33<00:00,  1.07it/s]
     43%|████▎     | 45/104 [1:10:07<1:32:14, 93.80s/it]

    [388240, 208630, 501900, 627430, 286520, 331520, 435520, 274280, 44200]
    Calculating log density
    0.0001440273833564078
    0.00014268140042359424


    100%|██████████| 100/100 [01:32<00:00,  1.08it/s]
     44%|████▍     | 46/104 [1:11:40<1:30:32, 93.66s/it]

    [582440, 647340, 616820, 83630, 295390, 157210, 590570, 485800, 466870]
    Calculating log density
    0.00014402738335640773
    0.00014313114222981948


    100%|██████████| 100/100 [01:33<00:00,  1.07it/s]
     45%|████▌     | 47/104 [1:13:14<1:29:02, 93.72s/it]

    [501560, 512460, 599910, 634120, 235840, 473360, 469710, 122890]
    Calculating log density
    0.00014402738335641215
    0.00014133508510191662


    100%|██████████| 100/100 [01:31<00:00,  1.10it/s]
     46%|████▌     | 48/104 [1:14:46<1:27:01, 93.25s/it]

    [63960, 559170, 376520, 269130, 466550, 532780, 459670, 591720, 490450]
    Calculating log density
    0.00014402738335640803
    0.0001423982834732577


    100%|██████████| 100/100 [01:33<00:00,  1.07it/s]
     47%|████▋     | 49/104 [1:16:21<1:25:47, 93.59s/it]

    [70770, 157290, 212270, 239960, 42800, 302720, 45620, 192000, 211210]
    Calculating log density
    0.00014402738335640838
    0.00014243003016392355


    100%|██████████| 100/100 [01:33<00:00,  1.07it/s]
     48%|████▊     | 50/104 [1:17:55<1:24:21, 93.74s/it]

    [372390, 413650, 84380, 627750, 583840, 519760, 585960, 217480, 389610]
    Calculating log density
    0.00014402738335640792
    0.00014276187398954782


    100%|██████████| 100/100 [01:32<00:00,  1.08it/s]
     49%|████▉     | 51/104 [1:19:28<1:22:43, 93.66s/it]

    [533920, 443710, 27360, 250960, 517500, 627700, 224060, 536320, 535280]
    Calculating log density
    0.0001440273833564081
    0.00014285274054082195


    100%|██████████| 100/100 [01:33<00:00,  1.07it/s]
     50%|█████     | 52/104 [1:21:02<1:21:10, 93.66s/it]

    [291410, 350000, 487550, 159330, 377350, 454780, 513930, 608920, 308320]
    Calculating log density
    0.0001440273833564077
    0.00014112297740783962


    100%|██████████| 100/100 [01:33<00:00,  1.07it/s]
     51%|█████     | 53/104 [1:22:36<1:19:46, 93.84s/it]

    [165050, 165500, 141040, 491770, 503480, 372300, 242690, 270650, 250000]
    Calculating log density
    0.00014402738335640778
    0.00014266746871930975


    100%|██████████| 100/100 [01:33<00:00,  1.07it/s]
     52%|█████▏    | 54/104 [1:24:10<1:18:10, 93.81s/it]

    [439860, 507840, 176120, 97490, 220060, 613030, 222630, 442360, 164380]
    Calculating log density
    0.000144027383356407
    0.00014209877117107942


    100%|██████████| 100/100 [01:34<00:00,  1.06it/s]
     53%|█████▎    | 55/104 [1:25:45<1:16:50, 94.08s/it]

    [465370, 466000, 465650, 164360, 223730, 536540]
    Calculating log density
    0.00014402738335640938
    0.00014064321344202472


      0%|          | 0/100 [00:00<?, ?it/s]
     53%|█████▎    | 55/104 [1:25:45<1:16:24, 93.55s/it]



    ---------------------------------------------------------------------------

    IndexError                                Traceback (most recent call last)

    Cell In[24], line 1
    ----> 1 vis.pose.sample_arena3D(
          2     pose_aligned,
          3     connectivity,
          4     labels=data_obj.data["Cluster"],
          5     n_samples=9,
          6     centered=True,
          7     VID_NAME = "cluster",
          8     N_FRAMES=100,
          9     fps=90,
         10     watershed=data_obj.ws,
         11     embed_vals=data_obj.embed_vals,
         12     filepath=config["out_path"],
         13 )


    File ~/working/neuroposelib/src/neuroposelib/visualization/pose.py:98, in sample.<locals>.wrapper(pose, connectivity, labels, VID_NAME, centered, n_samples, N_FRAMES, watershed, embed_vals, **kwargs)
         91     cat_watershed.watershed_map = np.where(
         92         watershed.watershed_map == cat, 1, 0.1
         93     )
         94     cat_watershed.watershed_map = np.where(
         95         watershed.watershed_map == 0, 0, cat_watershed.watershed_map
         96     )
    ---> 98 func(
         99     pose=pose[sampled_slice, ...],
        100     connectivity=connectivity,
        101     VID_NAME=VID_NAME + str(cat),
        102     embed_vals=cat_embed_vals,
        103     watershed=cat_watershed,
        104     n_samples=num_points,
        105     N_FRAMES=N_FRAMES,
        106     **kwargs,
        107 )


    File ~/working/neuroposelib/src/neuroposelib/visualization/pose.py:132, in sample_arena3D(pose, connectivity, n_samples, VID_NAME, N_FRAMES, watershed, embed_vals, filepath, **kwargs)
        129     else:
        130         density = watershed.watershed_map
    --> 132     arena3D_map(
        133         pose=pose,
        134         density=_mask_density(density, watershed.watershed_map, eps=EPS * 1.01),
        135         watershed_borders=watershed.borders,
        136         connectivity=connectivity,
        137         frames=np.arange(n_samples) * N_FRAMES,
        138         centered=False,
        139         N_FRAMES=N_FRAMES,
        140         VID_NAME=VID_NAME + ".mp4",
        141         SAVE_ROOT="".join([filepath, "/skeleton_vids/"]),
        142         **kwargs,
        143     )
        144 else:
        145     arena3D(
        146         pose=pose,
        147         connectivity=connectivity,
       (...)
        153         **kwargs,
        154     )


    File ~/working/neuroposelib/src/neuroposelib/visualization/pose.py:256, in arena3D_map(pose, density, watershed_borders, connectivity, frames, centered, N_FRAMES, fps, dpi, VID_NAME, SAVE_ROOT)
        254 for curr_frame in tqdm.tqdm(range(N_FRAMES)):
        255     curr_frames = curr_frame + np.arange(len(frames)) * N_FRAMES
    --> 256     ax_3d = _pose3D_arena(
        257         ax_3d, pose_3d, COLORS, links, curr_frames, limits, figsize
        258     )
        260     # grab frame and write to vid
        261     writer.grab_frame()


    File ~/working/neuroposelib/src/neuroposelib/visualization/pose.py:437, in _pose3D_arena(ax_3d, data, COLORS, links, frames, limits, size, title)
        435 (rows, cols) = size
        436 # import pdb; pdb.set_trace()
    --> 437 kpts_3d = np.reshape(data[frames, :, :], (len(frames) * data.shape[-2], 3))
        439 ax_3d = _pose3D_frame(
        440     ax_3d, kpts_3d, COLORS, links, limits  # , figsize=(cols * 5, rows * 5)
        441 )
        442 ax_3d.xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))


    IndexError: index 600 is out of bounds for axis 0 with size 600



    
![png](./tutorial_files/tutorial_50_115.png)
    

