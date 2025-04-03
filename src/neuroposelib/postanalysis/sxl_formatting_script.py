"""
    This is a version of sxl_formatting that takes the form of a script which can be 
    called with relevant variable values and run.

    Modularizing this could help in using the functions as an API as well. 
"""

# from postanalysis.features import *
import dappy.DataStruct as ds
import dappy.visualization as vis
import numpy as np
import pandas as pd
from dappy import read, write
from dappy.embed import Watershed, Embed
import pickle
import sys
# import analysis

import re
from tqdm import tqdm

import matplotlib.pyplot as plt
# from scipy import stats
# from statsmodels import stats 
# needed to install statsmodels via pip install not conda install, could not import when using conda install

from matplotlib.colors import LogNorm
import seaborn as sns
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from scipy.spatial.distance import squareform

# from scipy.stats.contingency import expected_freq
from scipy.stats import chisquare

from skimage.segmentation import watershed
# from skimage import measure


# from postanalysis.SXL_protofunctions import *
import os

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

## Required Variables for this analysis:
analysis_key = "vps35_ASAPall_20240721"
load_suffix = "fromcross_test_tbsf_20241227.2"
data_object_path = '/hpc/group/tdunn/Bryan_Rigs/SmallOpenField/dappy/results/crosstest_try_tailbspinef_20241227.2/datastruct.p'
cluster_occupancy_path = "/hpc/group/tdunn/Bryan_Rigs/SmallOpenField/dappy/results/crosstest_try_tailbspinef_20241227.2/cluster_occupancy.csv"
figure_folder_name = "20250115_Plots"


# Load relevant data
paths = read.config("../../configs/path_configs/" + analysis_key + ".yaml")
# params = read.config("../configs/param_configs/fitsne_" + analysis_key + ".yaml")
params = read.config("../../configs/param_configs/fitsne_ASAPall_20240721.yaml")
connectivity = read.connectivity(
    path=paths["skeleton_path"], skeleton_name=paths["skeleton_name"]
)

if 'data_obj' in locals():
    print('data_obj already exists...')
else:
    # print("Reading data_obj from " + "".join([paths["out_path"], params["label"], "_", load_suffix, "/datastruct.p"]) + " ...")
    # data_obj = pickle.load(
    #     open("".join([paths["out_path"], params["label"], "_", load_suffix, "/datastruct.p"]), "rb")
    # )
    print("Reading data_obj from {}".format(data_object_path) + " ...")
    data_obj = pickle.load(open(data_object_path,"rb"))

# Read cluster occupancy from with pd.open_csv
if 'cluster_occupancy' in locals():
    print('cluster_occupancy already exists...')
else:
    print("Reading cluster_occupancy.cv from " + cluster_occupancy_path + " ...")
    cluster_occupancy = pd.read_csv(cluster_occupancy_path)


#############################################################
# Prepare data for statistical tests
#############################################################
df = data_obj.data
mf = data_obj.meta
mff = data_obj.meta_by_frame
# imported as "wide/record" table format
cf = pd.read_csv(cluster_occupancy_path, index_col=0)
cf.columns = cf.columns.values.astype(float).astype(int) # need to convert string (stored as "object") to float, then integer, error when trying to convert ot integer directly
cf = cf.rename_axis("Clusters", axis="index")
cf = cf.rename_axis("id", axis="columns") # have to do this haver redefining columns if not will lose the name

df.id = df.id.astype(int)


df_Original = df.copy()
mf_Original = mf.copy()
mff_Original = mff.copy()

dfs_to_map = [df, mf, mff]
col_rename = {"Condition": "ConditionName", "Timepoint": "Condition"}

for idx,df_ in enumerate(dfs_to_map):
    print(idx)
    map_to_group(df_, column_name="Timepoint", do_col_rename=col_rename)
    df_ = df_.rename(columns=col_rename, inplace=True)

# use map_to_group to do the following:

# df.loc[:,'GroupID'] = df.Condition
# df.loc[df.Condition=='AssayTest','GroupID'] = 1
# df.loc[df.Condition=='Control','GroupID'] = 2
# df.loc[df.Condition=='LPS083','GroupID'] = 3
# df = df.rename(columns={"Condition": "ConditionName", "Timepoint": "Condition"})


# drop repeat video
# df = df.loc[~np.logical_and(df.AnimalID=='1691485_left',df.Date==20240627),:]
# mf = mf.loc[~np.logical_and(mf.AnimalID=='1691485_left',mf.Date==20240627),:]
# mff = mff.loc[~np.logical_and(mff.AnimalID=='1691485_left',mff.Date==20240627),:]

# drop animals with missing videos
# df = df.loc[np.logical_not(df.AnimalID=='1686940_left'),:]
# mf = mf.loc[np.logical_not(mf.AnimalID=='1686940_left'),:]
# mff = mff.loc[np.logical_not(mff.AnimalID=='1686940_left'),:]

# # drop animals with missing videos
# df = df.loc[np.logical_not(df.AnimalID=='1686941_both'),:]
# mf = mf.loc[np.logical_not(mf.AnimalID=='1686941_both'),:]
# mff = mff.loc[np.logical_not(mff.AnimalID=='1686941_both'),:]

# # drop animals with missing videos
# df = df.loc[np.logical_not(df.AnimalID=='1686941_none'),:]
# mf = mf.loc[np.logical_not(mf.AnimalID=='1686941_none'),:]
# mff = mff.loc[np.logical_not(mff.AnimalID=='1686941_none'),:]

# drop animals with missing videos
# df = df.loc[np.logical_not(df.GroupID==1),:]
# mf = mf.loc[np.logical_not(mf.GroupID==1),:]
# mff = mff.loc[np.logical_not(mff.GroupID==1),:]


# with pd.option_context('display.max_columns', 10,'display.max_colwidth', 1000):
#     print(mf.loc[[*range(62,72),*range(99,109)]])

# # double checked that first row is simply all zero
# if not np.any(cf.loc[0]!=0):
#     cf = cf.drop(index=0)

group = mf.loc[:,['id', 'GroupID', 'Condition', 'AnimalID']]
# #############################################################
# # Prepare Occupancy table
# #############################################################

# convert to "long/"stacked" format, becomes a series
occupancy = cf.stack() # ? same as set_index if only a 
occupancy = occupancy.to_frame('occupancy') # need to convert to DataFrame to merge
occupancy = occupancy.reset_index() # have to have same index as datagraoup to be combined with

occupancy_by_group = pd.merge(occupancy, group, on='id', validate='many_to_one')
occupancy_by_group = occupancy_by_group.set_index(['Condition','Clusters',  'GroupID', 'id']).sort_index()

# create cluster occupancies for combined Habituation and Baseline videos
og_all = occupancy_by_group.reset_index()
og_all = og_all.set_index('Condition')
# ogh = og_all.loc['Habituation'].reset_index(drop=True).set_index(['Clusters','GroupID','AnimalID'])
# ogb = og_all.loc['Baseline'].reset_index(drop=True).set_index(['Clusters','GroupID','AnimalID'])
# # ogt = og_all.loc['Treatment'].reset_index(drop=True).set_index(['Clusters','GroupID','AnimalID'])

# # add cluster frequencies for Habituation and Baseline
# oga = ogh.add(ogb)
# oga.occupancy=oga.occupancy/2 # normalize occupancy so that it sums to 1
# # oga.id = ogb.iloc[-1].id + 1 + ogh.id # create unique ID (id) for 'All" condition
# oga.id = ogt.iloc[-1].id + 1 + ogh.id # create unique ID (id) for 'All" condition
# oga.id = oga.id.astype(int)


# ogh.loc[:,'Condition'] = 'Habituation'
# ogb.loc[:,'Condition'] = 'Baseline'
# # oga.loc[:,'Condition'] = 'All'
# # ogt.loc[:,'Condition'] = 'Treatment'

# # # check that occupancy values all sum to 1 for each AnimalID
# # oga.loc[:,:,'a1'].occupancy.sum()
# # ogb.loc[:,:,'a1'].occupancy.sum()
# # ogh.loc[:,:,'a1'].occupancy.sum()



# og = pd.concat([ogh, ogb], axis=0)
# # og = pd.concat([ogh, ogb, oga, ogt], axis=0)
# # og = pd.concat([ogh, ogb, ogt], axis=0)
# # og = pd.concat([ogb, ogt], axis=0)

# Modified for ASTROn metadata
import pdb;pdb.set_trace()
# og1 = og_all.loc['Baseline'].reset_index(drop=True).set_index(['Clusters','GroupID','AnimalID'])
# og2 = og_all.loc['DART'].reset_index(drop=True).set_index(['Clusters','GroupID','AnimalID'])
# og3 = og_all.loc['NextDay'].reset_index(drop=True).set_index(['Clusters','GroupID','AnimalID'])
# ogt = og_all.loc['Treatment'].reset_index(drop=True).set_index(['Clusters','GroupID','AnimalID'])
# og1 = og_all.loc['Habituation'].reset_index(drop=True).set_index(['Clusters','GroupID','AnimalID'])
# og2 = og_all.loc['Baseline'].reset_index(drop=True).set_index(['Clusters','GroupID','AnimalID'])
# og3 = og_all.loc['Timepoint1'].reset_index(drop=True).set_index(['Clusters','GroupID','AnimalID'])
# og4 = og_all.loc['Timepoint1'].reset_index(drop=True).set_index(['Clusters','GroupID','AnimalID'])
og1 = og_all.loc['Habituation'].reset_index(drop=True).set_index(['Clusters','GroupID','AnimalID'])
og2 = og_all.loc['Demo'].reset_index(drop=True).set_index(['Clusters','GroupID','AnimalID'])


# og1.loc[:,'Condition'] = 'Baseline'
# og2.loc[:,'Condition'] = 'DART'
# og3.loc[:,'Condition'] = 'NextDay'
# og = pd.concat([og1, og2, og3], axis=0)
# og1.loc[:,'Condition'] = 'Habituation'
# og2.loc[:,'Condition'] = 'Baseline'
# og3.loc[:,'Condition'] = 'Timepoint1'
# og4.loc[:,'Condition'] = 'Timepoint2'
og1.loc[:,'Condition'] = 'Habituation'
og2.loc[:,'Condition'] = 'Demo'
# og = pd.concat([og1,og2,og3,og4], axis=0)
og = pd.concat([og1,og2], axis=0)

og = og.reset_index().set_index(['Condition', 'Clusters', 'GroupID', 'AnimalID', 'id']).sort_index(level=['Condition','Clusters', 'GroupID', 'id', 'AnimalID'])

# oga_pivot = og.loc['All',:].reset_index().pivot(index=['id'], columns='Clusters', values='occupancy')
# ogb_pivot = og.loc['Baseline',:].reset_index().pivot(index=['id'], columns='Clusters', values='occupancy')
# ogh_pivot = og.loc['Habituation',:].reset_index().pivot(index=['id'], columns='Clusters', values='occupancy')
# ogt_pivot = og.loc['Treatment',:].reset_index().pivot(index=['id'], columns='Clusters', values='occupancy')

##################################
# Parameters
##################################
parameter_suffix = load_suffix
# figure_folder_name = "20240418_Occupancy_ExcludeBadClusters"
# figure_folder_name = "20240726_Plots"

figure_folder_path = "".join([paths["out_path"], params["label"], "_", parameter_suffix, "/", figure_folder_name])
if not os.path.exists(figure_folder_path):
    os.makedirs(figure_folder_path)

##################################
# Get cluster occupancy heat map
# Export occupancy data for matlab analysis
##################################
og = og.reset_index(drop=False).set_index(['Condition', 'Clusters', 'GroupID', 'AnimalID', 'id']).sort_index(level=['Condition','Clusters', 'GroupID', 'id', 'AnimalID'])
og_pivot = og.reset_index(drop=False).pivot(index=['Condition', 'GroupID', 'AnimalID', 'id'], columns='Clusters', values='occupancy')
og_save_path = ''.join([paths['out_path'],params['label'],"_",parameter_suffix,'/og_pivot.csv'])
# if os.path.exists(og_save_path):
#     print("og_pivot.csv already exists at" + og_save_path)
# else:
#     print("Saving og_pivot.csv to " + og_save_path)
#     og_pivot.to_csv(og_save_path )
print("Saving og_pivot.csv to " + og_save_path)
og_pivot.to_csv(og_save_path )