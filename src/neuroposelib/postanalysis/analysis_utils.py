### Dappy Utilities
import neuroposelib.DataStruct as ds
import neuroposelib.visualization as vis
from neuroposelib import read, write
from neuroposelib import preprocess
from neuroposelib import write
from neuroposelib import features
from neuroposelib import analysis
from neuroposelib.embed import Watershed, Embed

import numpy as np
import pandas as pd
import pickle
import sys
import re
from tqdm import tqdm
import os
import copy
import time
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import seaborn as sns

from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from scipy.spatial.distance import squareform
import scipy.spatial.distance as distances
import scipy.stats as stats

# from scipy.stats.contingency import expected_freq
from scipy.stats import chisquare

from skimage.segmentation import watershed
# from skimage import measure

import dask



def map_to_group(
    inpdf,
    column_name='Condition', 
    map_dict={
        'Habituation': 1, 
        'Demo':2
        },
    do_col_rename = None
    ):
    """
        Does the following if all params are given:
        # df.loc[:,'GroupID'] = df.Condition
        # df.loc[df.Condition=='AssayTest','GroupID'] = 1
        # df.loc[df.Condition=='Control','GroupID'] = 2
        # df.loc[df.Condition=='LPS083','GroupID'] = 3
        # df = df.rename(columns={"Condition": "ConditionName", "Timepoint": "Condition"})
    """
    
    inpdf[column_name] = pd.Categorical(inpdf[column_name].astype(str))
    inpdf['GroupID'] = inpdf[column_name].cat.codes + 1
    
    if do_col_rename is not None:
        inpdf = inpdf.rename(columns=do_col_rename) 

def get_freq_df(config, file_name, idx_ax_name='Clusters', col_ax_name='id'):
    cf = pd.read_csv(''.join([config['out_path'],f'/{file_name}']), index_col=0)
    
    # if columns have non-string name, change them to int
    try:
        cf.columns = cf.columns.values.astype(int)
    except ValueError as ve1:
        if 'invalid literal for int() with base 10:' in str(ve1):
            cf.columns = cf.columns.values.astype(float).astype(int)
    
    cf = cf.rename_axis(idx_ax_name, axis="index")
    cf = cf.rename_axis(col_ax_name, axis="columns")

    return cf

def get_data_obj_from_file(
    config,
    file_name,
    col_renames = None,
    group_id_col = 'Timepoint',
    ):
    with open(''.join([config['out_path'], f"/{file_name}"]), "rb") as dobj:
        try:
            data_obj = pickle.load(dobj)
        except Exception as e:
            if 'dappy' in str(e):
                unpickler = RenamingUnpickler(dobj)
                dobj_1 = unpickler.load()
            else:
                raise e
    return process_data_obj(config, data_obj, col_renames, group_id_col), data_obj

    
def process_data_obj(
                    config,
                    data_obj,
                    col_renames = None,
                    group_id_col = 'Timepoint',
                    ):
    
    # try: 
    #     df = data_obj.data
    #     mf = data_obj.meta
    #     mff = data_obj.meta_by_frame
    # except AttributeError as ae:
    #     if "dict" in str(ae):
    #         df = data_obj['data']
    #         mf = data_obj['meta']
    #         mff = data_obj['meta_by_frame']
    
    if isinstance(data_obj, dict):
        df = data_obj['data']
        mf = data_obj['meta']
        mff = data_obj['meta_by_frame']
    else:
        df = data_obj.data
        mf = data_obj.meta
        mff = data_obj.meta_by_frame

    dfs_to_map = [df, mf, mff]
    if col_renames is not None:
        # col_rename = {"Condition": "ConditionName", "Timepoint": "Condition"}
        col_rename = col_renames

        for idx,df_ in enumerate(dfs_to_map):
            print(idx)
            map_to_group(df_, column_name=group_id_col, do_col_rename=col_rename)
            df_ = df_.rename(columns=col_rename, inplace=True)
    
    return df, mf, mff

def get_pivot_table(mf, freq_df, 
                    relevant_columns_frm_meta = ['Condition', 'GroupID', 'AnimalID', 'id'],
                    piv_col = 'Clusters', 
                    piv_vals = 'occupancy',
                    index_level = [ 'Condition',
                                    'Clusters', 
                                    'GroupID', 
                                    'id', 
                                    'AnimalID'],
                    ):

    group = mf.loc[:,relevant_columns_frm_meta]
    occu_by_group = pd.merge(freq_df.stack().to_frame(piv_vals).reset_index(), group, on='id', 
                                validate='many_to_one')
    pivot_idx = [col for col in index_level if col not in [piv_col, piv_vals] ]
    as_og = occu_by_group.set_index(index_level).sort_index(level=index_level)
    as_og_pivot = as_og.reset_index().pivot(index=pivot_idx, columns=piv_col, values=piv_vals)

    return as_og_pivot, pivot_idx

def centroid_distance(arr_1, arr_2, axis=0, distance_metric=distances.euclidean):
    # print(f"Shape of arr1 = {arr_1.shape}, Shape of arr2 = {arr_2.shape}")
    cent_1 = arr_1.mean(axis=axis)
    cent_2 = arr_2.mean(axis=axis)

    # print(f"Shape of cent1 = {cent_1.shape}, Shape of cent2 = {cent_2.shape}")
    # if len(cent_1.shape) > 1:
    # return distances.cdist(cent_1, cent_2)
    # else:
    return distances.euclidean(cent_1, cent_2)


@dask.delayed
def get_shuffled_stat(occu_data, len_gp_1, len_gp_2, distance_function):

        # Do a random permutation
        rand_perm = np.random.permutation(occu_data)
        gp_1 = rand_perm[:len_gp_1]
        gp_2 = rand_perm[len_gp_1:]

        len_gp_1 = gp_1.shape[0]
        len_gp_2 = gp_2.shape[0]

        shuffled_stat = distance_function(gp_1,gp_2)

        return shuffled_stat

def do_permutation_test(pivot_df, 
                        group_names, 
                        ndraws=None, 
                        distance_function=centroid_distance,
                        **kwargs2):
    """
    Lets say there are 2 groups, and 
        1. We label the mouse randomly
        2. Take the difference between the centroids of the 2 groups as test-statistic
        3. Plot the test statistic distribution, and check the p-value for the position of actual statistic
    
    Inputs:
        pivot_df: dataframe of the form generated by above script
                | Condition | GroupID | AnimalID | 0 | 1 | 2 | 3 | 4 |
                |---|---|---|---|---|---|---|---|
                | baseline | 1686940_both | 0 | 0.00 | 0.00 | 0.000139 | 0.006556 | 0.000000 |
                | baseline | 1686940_left | 0 | 0.00 | 0.00 | 0.000000 | 0.008056 | 0.000000 |
                | habituation | 21686940_both | 0 | 0.00 | 0.000306 | 0.002250 | 0.000000 | 0.000000 |
                | habituation | 21686940_none | 0 | 0.00 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
        group_names: List of condition names
        ndraws: 
    """
    if ndraws is None:
        ndraws=10000
    
    subsample_df = pivot_df.loc[group_names].reset_index()
    gp_1 = pivot_df.loc[group_names[0]]
    gp_2 = pivot_df.loc[group_names[1]]
    len_gp_1 = gp_1.shape[0]
    len_gp_2 = gp_2.shape[0]

    assert len_gp_1+len_gp_2 == subsample_df.shape[0]

    act_stat = distance_function(gp_1.values,gp_2.values)

    cols = subsample_df.columns
    cluster_cols = [col for col in cols if isinstance(col,int)]
    occupancy_data = subsample_df[cluster_cols]
    
    test_stat_dist = np.zeros(ndraws)
    delayed_results = []
    for draw in range(ndraws):

        # Do a random permutation
        # rand_perm = np.random.permutation(occupancy_data)
        # gp_1 = rand_perm[:len_gp_1]
        # gp_2 = rand_perm[len_gp_1:]

        # len_gp_1 = gp_1.shape[0]
        # len_gp_2 = gp_2.shape[0]

        # shuffled_stat = distance_function(gp_1,gp_2)

        # test_stat_dist[draw] = shuffled_stat

        # Try dask based parallelization
        shuffled_stat = get_shuffled_stat(occupancy_data, len_gp_1, len_gp_2, distance_function)
        delayed_results.append(shuffled_stat)

    test_stat_dist = np.asarray(dask.compute(*delayed_results))
    
    count_more_extreme = np.sum(test_stat_dist >= act_stat)
    p_value = count_more_extreme / len(test_stat_dist)

    return p_value, test_stat_dist, act_stat


def do_pairwise_tests(pvt_tab, 
                        ndraws=1000000, 
                        condition='Condition', 
                        ret_dists=False,
                        nbins=10
                    ):
    uniq_conds = pvt_tab.index.get_level_values(condition).astype(str).unique()
    num_unq_conds = len(uniq_conds)
    heatmap = np.zeros((num_unq_conds, num_unq_conds))
    if ret_dists:
        stat_dists = np.zeros((num_unq_conds, num_unq_conds, 2, nbins+1))
        act_stats = np.zeros((num_unq_conds, num_unq_conds))
    

    for idx1, cond1 in enumerate(uniq_conds):
        for idx2, cond2 in enumerate(uniq_conds):
            p_val, stat_dist, act_stat = do_permutation_test(
                                                    pvt_tab, 
                                                    [cond1, cond2], 
                                                    ndraws=ndraws)
            heatmap[idx1,idx2] = p_val
            if ret_dists:
                st_hist, st_bins = np.histogram(stat_dist)
                stat_dists[idx1,idx2,0,1:] = st_hist
                stat_dists[idx1,idx2,1,:] = st_bins
                act_stats[idx1,idx2] = act_stat

    if ret_dists:
        return heatmap, stat_dists, act_stats
    else:
        return heatmap
    
class RenamingUnpickler(pickle.Unpickler):
    """
    Unpickler for backward compatibility with DANNCE
    """
    def find_class(self, module, name):
        # print(f'module = {module}, name = {name}')
        if 'dappy' in module:
            module = module.replace('dappy','neuroposelib')
        return super().find_class(module, name)

def read_datastruct(config):
    """
    Function to enable loading a datastruct (or any pickle file) containing 
    dappy based data types to neuroposelib based datatypes.
    For backward compatibility with dappy 
    """
    with open(config['out_path'] + "/datastruct.p", "rb") as file:
        unpickler = RenamingUnpickler(file)
        data_obj = unpickler.load()
        return data_obj
