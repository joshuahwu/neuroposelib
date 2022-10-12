import embed
import Python_analysis.engine.FeatureStruct as ds
import validation as val
import visualization as vis

import numpy as np
import pandas as np
import matplotlib.pyplot as plt

import pickle

an_dir = '../results/24_final/batch_fitsne_4_25/'
dstruct = pickle.load(open(''.join([an_dir,'datastruct.p']),'rb'))

embedder_k = embed.BatchEmbed().load_pickle(''.join([an_dir,'batch_embed.p']))
# import pdb; pdb.set_trace()
embedder_k.lr = embedder_k._lr
embedder_k.perplexity = embedder_k._perplexity
kf_k = val.KFoldEmbed(out_path = dstruct.out_path).run(embedder_k,
                                            param = 'k',
                                            param_range = [1,2,3,4,5,6,7,8,10,12,15,20,25,35,50])
kf_k.plot_error()

embedder_t = embed.BatchEmbed().load_pickle(''.join([an_dir,'batch_embed.p']))
embedder_t.transform_method = 'sklearn_rf'
embedder_t.lr = embedder_t._lr
embedder_t.perplexity = embedder_t._perplexity
kf_t = val.KFoldEmbed(out_path = dstruct.out_path).run(embedder_t,
                                            param = 'n_trees',
                                            param_range = [25,50,75,100,150,200,250,300,400,500,1000])

kf_t.plot_error()

import pdb; pdb.set_trace()