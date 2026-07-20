# neuroposelib - 3D Animal Pose Analysis in Python

Module for analysis of 3D animal pose sequences. Based on work by [Berman et al. (2014)](https://royalsocietypublishing.org/doi/full/10.1098/rsif.2014.0672) and [Marshall et al. (2020)](https://www.sciencedirect.com/science/article/pii/S0896627320308941).

## Installation

If you just want to use neuroposelib as a library, install it directly from GitHub:

```
pip install git+https://github.com/joshuahwu/neuroposelib.git
```

If you'd like UMAP support for behavioral embeddings, install the `umap` extra instead:

```
pip install "neuroposelib[umap] @ git+https://github.com/joshuahwu/neuroposelib.git"
```

If you want to run the tutorials or work on the source, clone the repository and install it in editable mode instead. Install the latest version of [Miniconda](https://docs.conda.io/en/latest/miniconda.html) on your machine, then:

```
git clone https://github.com/joshuahwu/neuroposelib.git
cd neuroposelib
conda env create -n neuroposelib python=3.11
conda activate neuroposelib
conda install -c conda-forge opentsne ffmpeg
pip install -e .
```

Note that `pip` and `setuptools` must be updated to the most recent versions.

To begin gaining familiarity with the functionality of this package, download the demo dataset at [this link](https://duke.box.com/v/demo-mouse-poses) or with the command line as follows:

```
cd neuroposelib
wget -v -O ./tutorials/demo_mouse.zip -L https://duke.box.com/shared/static/2ypagjda3gws3m0yqzzdrszb79yfznz8.zip
unzip ./tutorials/demo_mouse.zip -d ./tutorials/
rm ./tutorials/demo_mouse.zip
```
 and run through the code in `/tutorials/tutorial.ipynb`.

## Authors

- **Joshua Wu** - joshua.wu@duke.edu