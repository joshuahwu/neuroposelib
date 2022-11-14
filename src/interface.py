from DataStruct import DataStruct
import visualization as vis
from embed import Watershed, BatchEmbed, Embed
import tsnecuda as tc
import os
import hdf5storage
from sklearn.decomposition import PCA
import numpy as np
from typing import Optional, Union, List, Tuple
import yaml

def read_params_config(config_path: str):
    '''
    Reads in params_config
    '''
    with open(config_path) as f:
        config_dict = yaml.safe_load(f)

    return config_dict

def params_process(config_path: str):
    '''
    Loads in and processes params to fill in blanks
    '''
    params = read_params_config(config_path)

    if 'pca_features' not in params:
        params['pca_features'] = False

    print("Printing params")
    print(params)
    
    return params

def load_data(paths_config,
              pca_features: bool = False,
              filter_still: bool = False,
              downsample: int = 10):
    '''
    '''
    ds = DataStruct(config_path=paths_config)
    ds.load_feats(downsample=downsample)

    if filter_still:
        ds = filter_still_frames(ds)

    if pca_features:
        ds.features = PCA(n_components=60).fit_transform(ds.features)

    ds.load_meta()
    print("Final Size")
    print(ds.features.shape)

    return ds

def filter_still_frames(dstruct:DataStruct):
    # import pdb; pdb.set_trace()
    vel_sums = np.sum(np.abs(dstruct.features[:,60:79]), axis=1)
    filtered_struct = dstruct[vel_sums>4.25,:]
    return filtered_struct


def run_analysis(params_config: str,
                 ds: DataStruct):#,
                #  paths_config: str):

    params = params_process(params_config)

    # ds = load_data(paths_config,
    #                filter_still=params['filter_still'],
    #                pca_features=params['pca_features'],
    #                downsample=params['downsample'])

    ds.out_path = ''.join([ds.out_path,'/',params['label'],'/'])

    # ds = ds[ds.data['Condition'].isin(['Baseline','Lesion2','LDOPA']),:]

    if not os.path.exists(ds.out_path):
        os.makedirs(ds.out_path)

    if params['analysis'] == 'exp_embed':
        # Subset out conditions
        ds_exp = ds[ds.data[params['column']].isin(params['conditions']).tolist(),:]
        ds_exp, embedder = embed_pipe(ds_exp, params, save_embedder = params['save_embedder'])

        # Embedding of select conditions
        watershed_cluster(ds_exp,
                          sigma = params['single_embed']['sigma'],
                          column = params['density_by_column'],
                          plots_label = "map_temp",
                          save_ws = True)
        ds_exp.write_pickle(out_path=''.join([ds_exp.out_path,"/exp_"]))

        ds.embed_vals = embedder.predict(ds.features) # Embed entire dataset
    elif params['analysis'] == 'embed':
        ds, embedder = embed_pipe(ds, params, save_embedder = params['save_embedder'])

    ds = watershed_cluster(ds,
                           sigma = params['transform_embed']['sigma'],
                           column = params['density_by_column'],
                           plots_label = "final",
                           save_ws = True)
    
    # vis.scatter_on_watershed(data=ds, watershed=ds.ws, column='Cluster')

    ds.write_pickle()
    if params['skeleton_vids']:
        # ds.pose_path = '/home/exx/Desktop/GitHub/CAPTURE_demo/CAPTURE_data/48_tadross_data/markers_preproc_rotated.mat'
        vis.skeleton_vid3D_cat(ds, 'Cluster', n_skeletons=10)

    return ds


def embed_pipe(dstruct: DataStruct,
               params: dict,
               save_embedder: bool = True):
    '''

    '''
    transform_params = params['transform_embed'].copy()
    transform_params['transform_method'] = transform_params.pop('method')
    transform_params.pop('sigma')

    embed_params = params['single_embed'].copy()
    embed_params.pop('sigma')

    if 'batch_embed' not in params: # Single embedding no batching
        embed_params['embed_method'] = embed_params.pop('method')
        embed_params.update(transform_params)
        # import pdb; pdb.set_trace()

        embedder = Embed(**embed_params)
        dstruct.embed_vals = embedder.embed(dstruct.features, save_self=True)

    elif 'batch_embed' in params: # Creates batchmaps and templates
        batch_params = params['batch_embed'].copy()
        batch_params['batch_method'] = batch_params.pop('method')
        batch_params.update(transform_params)
        batch_params['embed_method'] = embed_params['method']

        if params['load_embedder'] is None:
            embedder = BatchEmbed(**batch_params)
            embedder = embedder.fit(data = dstruct.features, 
                                    batch_id = dstruct.exp_id,
                                    save_batchmaps = dstruct.out_path,
                                    embed_temp = False)
        else:
            embedder = BatchEmbed().load_pickle(params['load_embedder'])
        # import pdb; pdb.set_trace()
        embedder.embed(**embed_params, save_self=True) # Embed template with chosen params

        dstruct.embed_vals = embedder.predict(dstruct.features) # Embed all input points
        # Cluster and plot densities of the template itself
        temp_struct = dstruct[embedder.temp_idx,:]
        watershed_cluster(temp_struct,
                          sigma = params['single_embed']['sigma'],
                          column = params['density_by_column'],
                          plots_label = "batch_temp")
 
        if save_embedder:
            embedder.save_pickle(dstruct.out_path)

    return dstruct, embedder

def watershed_cluster(dstruct: DataStruct,
                      sigma: int = 15,
                      column: Union[str,List[str]] = ['Condition'],
                      plots_label: str = '',
                      save_ws: bool = False):

    # Calculating watershed and densities of template to compare with 
    ws = Watershed(sigma = sigma,
                    max_clip = 1,
                    log_out = True,
                    pad_factor = 0.05)
    dstruct.data.loc[:,'Cluster'] = ws.fit_predict(data = dstruct.embed_vals)

    if save_ws:
        dstruct.ws = ws

    vis.density(ws.density, ws.borders,
                filepath = ''.join([dstruct.out_path,plots_label,'_density.png']))
    vis.scatter(dstruct.embed_vals, filepath=''.join([dstruct.out_path, plots_label,'_scatter.png']))

    for cat in column:
        vis.density_cat(data=dstruct, column=cat, watershed=ws, n_col=4,
                        filepath = ''.join([dstruct.out_path,plots_label,'_density_',cat,'.png']))

    return dstruct


