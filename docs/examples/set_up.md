## Setting Up `neuroposelib` For You
Joshua Wu

Duke University Biomedical Engineering

[Timothy Dunn Lab](https://www.tdunnlab.org/)

11 June, 2024

To follow this notebook, please download the contents of the [demo dataset](https://duke.box.com/v/demo-mouse-poses) into the `/neuroposelib/tutorials/demo_mouse/` directory.

General information regarding your `neuroposelib` analysis (e.g. file paths and hyperparameters) is contained in a `config.yaml` file.


```python
from neuroposelib import read

analysis_key = "tutorial"
config = read.config("../configs/" + analysis_key + ".yaml")
```

The core components to keep track of in `neuroposelib` are your metadata, your skeleton, your poses, and your session ids.

Let's first load in the poses and session ids.


```python
pose, ids = read.pose_h5(config["data_path"] + "demo_mouse.h5")
```

`pose` is a NumPy array containing your pose predictions of shape `(# frames, # keypoints, 3 Cartesian coordinates)`.

`ids` is an `int` iterable of length, `# frames`, which keeps track of the recording session each frame belongs to.


```python
print(pose.shape)
print(ids)
```

    (648000, 18, 3)
    [0 0 0 ... 1 1 1]


The first thing to set up is your `meta.csv` (called `demo_meta.csv` in this tutorial).

Each row of `demo_meta.csv` contains categorial information for a recording session (e.g. animal identity, sex, strain, experimental condition, etc.) and a session id. The specific categories are fully-determined by you, and depends on the behavioral comparisons you'd like to make.

Here the `meta` variable exactly recapitulates your `meta.csv` file in a `pandas` DataFrame.


```python
meta, meta_by_frame = read.meta(config["data_path"] + "demo_meta.csv", id=ids)

print(meta)
```

       id AnimalID     Sex       Strain Condition                     Path
    0   0       A0    Male  Adora2a-Cre  Baseline  ./data/demo_mouse_0.mat
    1   1       A1  Female  Adora2a-Cre  Baseline  ./data/demo_mouse_1.mat


`meta_by_frame` expands `meta` to assign your metadata to each frame of your pose array. It is also a `pandas` DataFrame and has rows equal to the total number of frames in your pose array. This may be useful in situations where you want to associate frame-wise continuous variables to your metadata.


```python
print(meta_by_frame.shape[0] == pose.shape[0])
```

    True


Finally, it is crucial to set up the connectivity of your keypoint representation. The `Connectivity` class in `neuroposelib` contains attributes for joint names and links between joints (i.e. segments). It also allows you to assign colors to the keypoints and segments for later plotting.

Follow the example in `configs/skeletons/demo_mouse.yaml` to create your own.


```python
connectivity = read.connectivity_config(
    path=config["skeleton_path"]
)
```

Note that the labels for the joints should be in order corresponding to the indices of your `pose` array.


```python
print(connectivity.joint_names)
```

    ['Snout', 'EarR', 'EarL', 'SpineF', 'SpineM', 'Tail_base', 'ForepawR', 'WristR', 'ElbowR', 'ForepawL', 'WristL', 'ElbowL', 'HindpawR', 'AnkleR', 'KneeR', 'HindpawL', 'AnkleL', 'KneeL']


If you choose to use Euler angle representations for your joints, you can also specify them in the connectivity config file as tuples of joints.


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


#### Using `neuroposelib` with [DANNCE](https://github.com/spoonsso/dannce)

If you're generating your 3D poses from DANNCE, loading into `neuroposelib` is straightforward.

Given DANNCE output paths to each recording session in the `meta.csv`, we can use `neuroposelib.read.pose_from_meta` to load and merge all recording sessions.


```python
pose, ids, meta, meta_by_frame = read.pose_from_meta(
    path=config["data_path"] + "demo_meta.csv",
    connectivity=connectivity,
    key="Path",
    file_type="dannce",
)

print(meta)
```

    2it [00:00,  2.71it/s]

       id AnimalID     Sex       Strain Condition                     Path
    0   0       A0    Male  Adora2a-Cre  Baseline  ./data/demo_mouse_0.mat
    1   1       A1  Female  Adora2a-Cre  Baseline  ./data/demo_mouse_1.mat


    



```python
print(pose.shape)
```

    (648000, 18, 3)

