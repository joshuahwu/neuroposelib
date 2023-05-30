import keypoint_moseq as kpms
import read
import numpy as np
import features
from jax_moseq.models import arhmm
from jax_moseq.utils import batch, unbatch
from keypoint_moseq.util import reindex_by_frequency
import jax
import jax.random as jr
import matplotlib.pyplot as plt
from tqdm.auto import trange

def plot_ll(key, ll_history, path: str):
    plt.title(f'Log Likelihood of {key}')
    plt.xlabel('Iteration')
    plt.ylabel('Log Likelihood')
    plt.plot(ll_history)
    plt.show()
    plt.savefig(path+'/ll_'+str(key)+'.png')

analysis_key = "ensemble_healthy"
paths = read.config("../configs/path_configs/" + analysis_key + ".yaml")

config = lambda: kpms.load_config("../configs/kpms_configs/")

connectivity = read.connectivity(
        path=paths["skeleton_path"], skeleton_name=paths["skeleton_name"]
    )

pose, id = read.pose_h5(paths["data_path"] + "pose_aligned.h5", dtype=np.float64)

# pose = features.rotate_spine(features.center_spine(pose), dtype=np.float64)

pca_feats, labels = read.features_h5(paths["out_path"] + "pca_feats.h5", dtype=np.float64)

# coordinates = {}
pca_data = {}
for i in np.unique(id):
    pca_data[str(i)] = pca_feats[id[::2]==i, :16] # 16 is the number of non-wavelet PCs
    # coordinates[str(i)] = pose[id==i, ...]

# ### PCA from keypoint moseq on just the coordinate positions
# data, labels = kpms.format_data(coordinates, **config())
# pca = kpms.fit_pca(**data,**config())
# kpms.plot_pcs(pca, project_dir=paths["out_path"], **config())
# kpms_model = kpms.init_model(data, pca=pca, **config())
# confidences = {key: np.ones_like(coordinates[key][...,0]) for key in sorted(pca_data.keys())}

## Wrangling own PCA data into correct format
Y,mask,labels = batch(pca_data, seg_length=config()['seg_length'], keys=sorted(pca_data.keys()))
Y = Y.astype(float)
pca_data_formatted = jax.device_put({'mask':mask, 'x':Y})#, 'conf':confidences})

## Initialize ARHMM model
hyper_params = arhmm.init_hyperparams(trans_hypparams=config()['trans_hypparams'],ar_hypparams=config()['ar_hypparams'])
model = arhmm.init_model(pca_data_formatted,hypparams=hyper_params, verbose=True)

## Fit states of ARHMM model
num_iters = 50
ll_keys = ['z', 'x']
ll_history = {key: [] for key in ll_keys}
for i in trange(num_iters):
    model = arhmm.resample_model(pca_data_formatted, **model)

    ll = arhmm.model_likelihood(pca_data_formatted, **model)
    for key in ll_keys:
        ll_history[key].append(ll[key].item())

for k, v in ll_history.items():
    plot_ll(k, v, paths["out_path"])

kpms.extract_results()

## Extract cluster/syllable labels
nlags = pca_data_formatted['x'].shape[1] - model['states']['z'].shape[1]
assert nlags == config()['ar_hypparams']['nlags']
lagged_labels = [(key,start+nlags,end) for key,start,end in labels]
syllables = unbatch(model['states']['z'], lagged_labels)
syllables = {k: np.pad(z[nlags:], (nlags,0), mode='edge') for k,z in syllables.items()}
syllables_reindexed = reindex_by_frequency(syllables)

import pdb; pdb.set_trace()

