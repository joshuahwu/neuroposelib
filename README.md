# 3D Animal Pose Analysis in Python

Module for analysis of 3D animal pose sequences. Based on work by [Berman et al. (2014)](https://royalsocietypublishing.org/doi/full/10.1098/rsif.2014.0672) and [Marshall et al. (2020)](https://www.sciencedirect.com/science/article/pii/S0896627320308941).

## Installation

Install the latest version of [Miniconda](https://docs.conda.io/en/latest/miniconda.html) on your machine.

The following steps will clone this repository, set up your conda environment, and install dappy.

Use `environment.yml` if you're on a Linux machine, and `environment_osx.yml` for Mac.

```
git clone https://github.com/joshuahwu/dappy.git
cd dappy
conda env create -n neuroposelib -f environment.yml
conda activate neuroposelib
conda install -c conda-forge opentsne
pip install -e .
```

Note that `pip` and `setuptools` must be updated to the most recent versions.

To begin gaining familiarity with the functionality of this package, download the demo dataset at [this link](https://duke.box.com/v/demo-mouse-poses) or with the command line as follows:

```
cd dappy
wget -v -O ./tutorials/data/demo_mouse.h5 -L https://duke.box.com/shared/static/zprn76pl31a9u1pp6gvxbmehn7p9zmbx.h5 
```
 and run through the code in `/tutorials/tutorial.ipynb`.

## Authors

- **Joshua Wu** - joshua.wu@duke.edu
