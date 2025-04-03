from features import *
import dappy.DataStruct as ds
import dappy.visualization as vis
import numpy as np
import pandas as pd
from dappy import read, write
from dappy.embed import Watershed, Embed
import pickle
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


from SXL_protofunctions import *
import os


#############################################################
# Load data
#############################################################
analysis_key = "vps35_ASAPall_20240721"
paths = read.config("../../configs/path_configs/" + analysis_key + ".yaml")
# params = read.config("../configs/param_configs/fitsne_" + analysis_key + ".yaml")
params = read.config("../../configs/param_configs/fitsne_ASAPall_20240721.yaml")
connectivity = read.connectivity(
    path=paths["skeleton_path"], skeleton_name=paths["skeleton_name"]
)

load_suffix = "fromtry3_copy"

# Read data object from pickle
if 'data_obj' in locals():
    print('data_obj already exists...')
else:
    # print("Reading data_obj from " + "".join([paths["out_path"], params["label"], "_", load_suffix, "/datastruct.p"]) + " ...")
    # data_obj = pickle.load(
    #     open("".join([paths["out_path"], params["label"], "_", load_suffix, "/datastruct.p"]), "rb")
    # )
    print("Reading data_obj from /hpc/group/tdunn/Bryan_Rigs/SmallOpenField/dappy/results/dune_try3/datastruct.p" + " ...")
    data_obj = pickle.load(open("/hpc/group/tdunn/Bryan_Rigs/SmallOpenField/dappy/results/dune_try3/datastruct.p","rb"))

# Read cluster occupancy from with pd.open_csv
if 'cluster_occupancy' in locals():
    print('cluster_occupancy already exists...')
else:
    print("Reading cluster_occupancy.cv from " + "/hpc/group/tdunn/Bryan_Rigs/SmallOpenField/dappy/results/dune_try3/cluster_occupancy.csv" + " ...")
    cluster_occupancy = pd.read_csv("/hpc/group/tdunn/Bryan_Rigs/SmallOpenField/dappy/results/dune_try3/cluster_occupancy.csv")




#############################################################
# Prepare data for statistical tests
#############################################################
df = data_obj.data
mf = data_obj.meta
mff = data_obj.meta_by_frame
# imported as "wide/record" table format
cf = pd.read_csv("/hpc/group/tdunn/Bryan_Rigs/SmallOpenField/dappy/results/dune_try3/cluster_occupancy.csv", index_col=0)
cf.columns = cf.columns.values.astype(float).astype(int) # need to convert string (stored as "object") to float, then integer, error when trying to convert ot integer directly
cf = cf.rename_axis("Clusters", axis="index")
cf = cf.rename_axis("id", axis="columns") # have to do this haver redefining columns if not will lose the name

df.id = df.id.astype(int)


df_Original = df.copy()
mf_Original = mf.copy()
mff_Original = mff.copy()

df.loc[:,'GroupID'] = df.Condition
df.loc[df.Condition=='AssayTest','GroupID'] = 1
df.loc[df.Condition=='Control','GroupID'] = 2
df.loc[df.Condition=='LPS083','GroupID'] = 3
df = df.rename(columns={"Condition": "ConditionName", "Timepoint": "Condition"})

mf.loc[:,'GroupID'] = mf.Condition
mf.loc[mf.Condition=='AssayTest','GroupID'] = 1
mf.loc[mf.Condition=='Control','GroupID'] = 2
mf.loc[mf.Condition=='LPS083','GroupID'] = 3
mf = mf.rename(columns={"Condition": "ConditionName", "Timepoint": "Condition"})

mff.loc[:,'GroupID'] = mff.Condition
mff.loc[mff.Condition=='AssayTest','GroupID'] = 1
mff.loc[mff.Condition=='Control','GroupID'] = 2
mff.loc[mff.Condition=='LPS083','GroupID'] = 3
mff = mff.rename(columns={"Condition": "ConditionName", "Timepoint": "Condition"})


# drop repeat video
df = df.loc[~np.logical_and(df.AnimalID=='1691485_left',df.Date==20240627),:]
mf = mf.loc[~np.logical_and(mf.AnimalID=='1691485_left',mf.Date==20240627),:]
mff = mff.loc[~np.logical_and(mff.AnimalID=='1691485_left',mff.Date==20240627),:]

# drop animals with missing videos
df = df.loc[np.logical_not(df.AnimalID=='1686940_left'),:]
mf = mf.loc[np.logical_not(mf.AnimalID=='1686940_left'),:]
mff = mff.loc[np.logical_not(mff.AnimalID=='1686940_left'),:]

# # drop animals with missing videos
# df = df.loc[np.logical_not(df.AnimalID=='1686941_both'),:]
# mf = mf.loc[np.logical_not(mf.AnimalID=='1686941_both'),:]
# mff = mff.loc[np.logical_not(mff.AnimalID=='1686941_both'),:]

# # drop animals with missing videos
# df = df.loc[np.logical_not(df.AnimalID=='1686941_none'),:]
# mf = mf.loc[np.logical_not(mf.AnimalID=='1686941_none'),:]
# mff = mff.loc[np.logical_not(mff.AnimalID=='1686941_none'),:]

# drop animals with missing videos
df = df.loc[np.logical_not(df.GroupID==1),:]
mf = mf.loc[np.logical_not(mf.GroupID==1),:]
mff = mff.loc[np.logical_not(mff.GroupID==1),:]


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
# og1 = og_all.loc['Baseline'].reset_index(drop=True).set_index(['Clusters','GroupID','AnimalID'])
# og2 = og_all.loc['DART'].reset_index(drop=True).set_index(['Clusters','GroupID','AnimalID'])
# og3 = og_all.loc['NextDay'].reset_index(drop=True).set_index(['Clusters','GroupID','AnimalID'])
# ogt = og_all.loc['Treatment'].reset_index(drop=True).set_index(['Clusters','GroupID','AnimalID'])
og1 = og_all.loc['Habituation'].reset_index(drop=True).set_index(['Clusters','GroupID','AnimalID'])
og2 = og_all.loc['Baseline'].reset_index(drop=True).set_index(['Clusters','GroupID','AnimalID'])
og3 = og_all.loc['Timepoint1'].reset_index(drop=True).set_index(['Clusters','GroupID','AnimalID'])
og4 = og_all.loc['Timepoint1'].reset_index(drop=True).set_index(['Clusters','GroupID','AnimalID'])

# og1.loc[:,'Condition'] = 'Baseline'
# og2.loc[:,'Condition'] = 'DART'
# og3.loc[:,'Condition'] = 'NextDay'
# og = pd.concat([og1, og2, og3], axis=0)
og1.loc[:,'Condition'] = 'Habituation'
og2.loc[:,'Condition'] = 'Baseline'
og3.loc[:,'Condition'] = 'Timepoint1'
og4.loc[:,'Condition'] = 'Timepoint2'
og = pd.concat([og1,og2,og3,og4], axis=0)

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
figure_folder_name = "20240726_Plots"

figure_folder_path = "".join([paths["out_path"], params["label"], "_", parameter_suffix, "/", figure_folder_name])
if not os.path.exists(figure_folder_path):
    os.makedirs(figure_folder_path)

##################################
# Get cluster occupancy heat map
# Export occupancy data for matlab analysis
##################################
og = og.reset_index(drop=False).set_index(['Condition', 'Clusters', 'GroupID', 'AnimalID', 'id']).sort_index(level=['Condition','Clusters', 'GroupID', 'id', 'AnimalID'])
og_pivot = og.reset_index(drop=False).pivot(index=['Condition', 'GroupID', 'AnimalID'], columns='Clusters', values='occupancy')
og_save_path = ''.join([paths['out_path'],params['label'],"_",parameter_suffix,'/og_pivot.csv'])
# if os.path.exists(og_save_path):
#     print("og_pivot.csv already exists at" + og_save_path)
# else:
#     print("Saving og_pivot.csv to " + og_save_path)
#     og_pivot.to_csv(og_save_path )
print("Saving og_pivot.csv to " + og_save_path)
og_pivot.to_csv(og_save_path )


#######################################
# Updated 20240625, SXL
#######################################
############################################
# Plot tSNE (with Padding) (Individual animals)
############################################
hf = df.loc[:, ['AnimalID', 'Condition', 'GroupID', 'frame', 'embed_vals']].set_index(['Condition', 'GroupID', 'AnimalID', 'frame'])
embed_dict = []
for animal_index, animal_id in enumerate(tqdm(np.sort(df.AnimalID.unique()))):
    for condition_index, condition in enumerate(tqdm(np.append(np.sort(df.Condition.unique()),'All'))):
    
    # for condition_index, condition in enumerate(["Baseline"]):
    
        if condition=='All':
            embed_vals = hf.loc[(slice(None), slice(None), animal_id, slice(None)), 'embed_vals']
        else:
            embed_vals = hf.loc[(condition, slice(None), animal_id, slice(None)), 'embed_vals']

        ## Old code does not account for padding  
        # n_bins = 1000
        # hist, xedges, yedges = np.histogram2d(
        #     [embed_vals.iloc[i][0] for i in range(np.size(embed_vals))],
        #     [embed_vals.iloc[i][1] for i in range(np.size(embed_vals))],
        #     bins=[n_bins, n_bins],
        #     density=False,
        # )

        data = np.array(embed_vals.values.tolist())   
        pad_factor = 0.05
        # hist_range = None
        # use hist_range from data_obj
        hist_range = data_obj.ws.hist_range
        n_bins =1000

        range_len = (
            np.ceil(np.amax(data, axis=0)) - np.floor(np.amin(data, axis=0))
        ).astype(int)
        padding = range_len * pad_factor

        # Calculate x and y limits for histogram and density
        if hist_range is None:
            print("Calculating new histogram ranges")
            hist_range = [
                [np.amin(data[:, 0]) - padding[0], np.amax(data[:, 0]) + padding[0]],
                [np.amin(data[:, 1]) - padding[1], np.amax(data[:, 1]) + padding[1]],
            ]
            # self.hist_range = [[int(np.floor(np.amin(data[:,0]))-padding[0]),int(np.ceil(np.amax(data[:,0]))+padding[0])],
            #                    [int(np.floor(np.amin(data[:,1]))-padding[1]),int(np.ceil(np.amax(data[:,1]))+padding[1])]]

        hist, xedges, yedges = np.histogram2d(
            data[:, 0],
            data[:, 1],
            bins=[n_bins, n_bins],
            range=hist_range,
            density=False,
        )
        
        hist = np.rot90(hist)
        density = hist/np.sum(hist)
        embed_dict.append({'Condition': condition, 'AnimalID': animal_id, 'density': density})

ha = pd.DataFrame.from_dict(embed_dict).set_index(['Condition', 'AnimalID'])

############################################
# Plot tSNE (with Padding)
############################################
hf = df.loc[:, ['AnimalID', 'Condition', 'GroupID', 'frame', 'embed_vals']].set_index(['Condition', 'GroupID', 'AnimalID', 'frame'])
embed_dict = []
for group_index, group_id in enumerate(tqdm(np.sort(df.GroupID.unique()))):
    for condition_index, condition in enumerate(tqdm(np.append(np.sort(df.Condition.unique()),'All'))):
    # for condition_index, condition in enumerate(["Baseline"]):
    
        if condition=='All':
            embed_vals = hf.loc[(slice(None), group_id, slice(None), slice(None)), 'embed_vals']
        else:
            embed_vals = hf.loc[(condition, group_id, slice(None), slice(None)), 'embed_vals']

        ## Old code does not account for padding  
        # n_bins = 1000
        # hist, xedges, yedges = np.histogram2d(
        #     [embed_vals.iloc[i][0] for i in range(np.size(embed_vals))],
        #     [embed_vals.iloc[i][1] for i in range(np.size(embed_vals))],
        #     bins=[n_bins, n_bins],
        #     density=False,
        # )

        data = np.array(embed_vals.values.tolist())   
        pad_factor = 0.05
        # hist_range = None
        # use hist_range from data_obj
        hist_range = data_obj.ws.hist_range
        n_bins =1000

        range_len = (
            np.ceil(np.amax(data, axis=0)) - np.floor(np.amin(data, axis=0))
        ).astype(int)
        padding = range_len * pad_factor

        # Calculate x and y limits for histogram and density
        if hist_range is None:
            print("Calculating new histogram ranges")
            hist_range = [
                [np.amin(data[:, 0]) - padding[0], np.amax(data[:, 0]) + padding[0]],
                [np.amin(data[:, 1]) - padding[1], np.amax(data[:, 1]) + padding[1]],
            ]
            # self.hist_range = [[int(np.floor(np.amin(data[:,0]))-padding[0]),int(np.ceil(np.amax(data[:,0]))+padding[0])],
            #                    [int(np.floor(np.amin(data[:,1]))-padding[1]),int(np.ceil(np.amax(data[:,1]))+padding[1])]]

        hist, xedges, yedges = np.histogram2d(
            data[:, 0],
            data[:, 1],
            bins=[n_bins, n_bins],
            range=hist_range,
            density=False,
        )
        
        hist = np.rot90(hist)
        density = hist/np.sum(hist)
        embed_dict.append({'Condition': condition, 'GroupID': group_id, 'density': density})

hg = pd.DataFrame.from_dict(embed_dict).set_index(['Condition', 'GroupID'])

###########################################################
# Introduce "Z"-Score (2024/04/16)
# Get mean density score for each cluster by animal from df.Cluster
###########################################################
zf = df.loc[:, ['AnimalID', 'Condition', 'frame', 'embed_vals', 'Cluster', 'GroupID']].set_index(['Condition', 'AnimalID', 'frame'])
cluster_dict = []
for condition_index, condition in enumerate(tqdm(np.append(np.sort(df.Condition.unique()),'All'))):
    for animal_index, animal_id in enumerate(tqdm(np.sort(df.AnimalID.unique()))):
        if condition=='All':
            clusters = zf.loc[(slice(None), animal_id, slice(None)), 'Cluster']
            group_id = int(np.unique(zf.loc[(slice(None), animal_id, slice(None)), 'GroupID']))
        else:
            clusters = zf.loc[(condition, animal_id, slice(None)), 'Cluster']
            group_id = int(np.unique(zf.loc[(condition, animal_id, slice(None)), 'GroupID']))
        total = np.size(clusters)
        for cluster_index, cluster_id in enumerate(tqdm(np.sort(df.Cluster.unique()))):
            counts = np.sum(clusters==cluster_id)
            density = counts/total
            cluster_dict.append({'Cluster': cluster_id,'Condition': condition, 'GroupID': group_id, 'AnimalID': animal_id, 'density': density, 'counts': counts, 'total': total})

zfc = pd.DataFrame.from_dict(cluster_dict)
zgc = pd.DataFrame.from_dict(cluster_dict).set_index(['Cluster', 'Condition', 'GroupID', 'AnimalID'])    
meanZ = zfc.groupby(['Cluster','Condition', 'GroupID']).mean()
sumZ = zfc.groupby(['Cluster','Condition', 'GroupID']).sum()
stdZ = zfc.groupby(['Cluster','Condition', 'GroupID']).std()

zgc.to_csv("".join([paths["out_path"], params["label"], "_", parameter_suffix, "/", figure_folder_name, "/zgc.csv"]))
meanZ.to_csv("".join([paths["out_path"], params["label"], "_", parameter_suffix, "/", figure_folder_name, "/meanZ.csv"]))
sumZ.to_csv("".join([paths["out_path"], params["label"], "_", parameter_suffix, "/", figure_folder_name, "/sumZ.csv"]))
stdZ.to_csv("".join([paths["out_path"], params["label"], "_", parameter_suffix, "/", figure_folder_name, "/stdZ.csv"]))

############################
# Calculate paired statistic
############################
std_dict = []
diffZ_dict = []
# 2024/06/30: found out the column for loop causes addition rows to be created
for group_index1, group_id1 in enumerate(np.sort(df.GroupID.unique())):
    for group_index2, group_id2 in enumerate(np.sort(df.GroupID.unique())):
        for condition_index, condition in enumerate(np.append(np.sort(df.Condition.unique()),'All')):
            for cluster_index, cluster_id in enumerate(tqdm(np.sort(df.Cluster.unique()))):
                for column_index, column_name in enumerate(zgc.columns):
                    std1 = stdZ.loc[(cluster_id, condition, group_id1),column_name]
                    if std1 == 0:
                            std1 = 1
                    std2 = stdZ.loc[(cluster_id, condition, group_id2),column_name]
                    if std2 == 0:
                            std2 = 1  
                    std_sum = np.sqrt(np.square(std1) + np.square(std2))  

                    if column_name == "density":
                        density_std = std_sum
                        # densityZ = z_score
                    if column_name == "counts":
                        counts_std = std_sum
                        # countsZ = z_score   
                    if column_name == "total":
                        total_std = std_sum
                        # countsZ = z_score                                                          

                    diff = sumZ.loc[(cluster_id, condition, group_id1),column_name] - sumZ.loc[(cluster_id, condition, group_id2),column_name]
                    if column_name == "density":
                        density = np.divide(diff,std_sum)
                        # densityZ = z_score
                    if column_name == "counts":
                        counts = np.divide(diff,std_sum)
                        # countsZ = z_score   
                    if column_name == "total":
                        total = np.divide(diff,std_sum)
                        # countsZ = z_score                                                          
                
                std_dict.append({'Cluster': cluster_id,'Condition': condition, 'GroupID1': group_id1, 'GroupID2': group_id2, 'density': density_std, 'counts': counts_std, 'total': total_std})
                diffZ_dict.append({'Cluster': cluster_id,'Condition': condition, 'GroupID1': group_id1, 'GroupID2': group_id2, 'density': density, 'counts': counts, 'total': total})

stdZpair = pd.DataFrame.from_dict(std_dict).set_index(['Cluster','Condition', 'GroupID1', 'GroupID2'])
stdZpair.to_csv("".join([paths["out_path"], params["label"], "_",parameter_suffix, "/", figure_folder_name, "/stdZpair.csv"]))

diffZpair = pd.DataFrame.from_dict(diffZ_dict).set_index(['Cluster','Condition', 'GroupID1', 'GroupID2'])
diffZpair.to_csv("".join([paths["out_path"], params["label"], "_",parameter_suffix, "/", figure_folder_name, "/diffZpair.csv"]))

diffZpair_sorted = diffZpair.loc[(slice(None),slice(None),slice(None),slice(None),),:].reset_index().sort_values(by=['Condition', 'GroupID1', 'GroupID2','density'],axis=0, ascending=[True, True, True, False]).set_index(['Condition', 'GroupID1', 'GroupID2'])
diffZpair_sorted.to_csv("".join([paths["out_path"], params["label"], "_", parameter_suffix, "/", figure_folder_name, "/diffZpair_sorted.csv"]))

############################################
# Calculate tSNE for PD and WT by GroupID from embed_values with Cluster as a predictor
############################################
hfc = df.loc[:, ['AnimalID', 'Condition', 'GroupID', 'frame', 'embed_vals', 'Cluster']].set_index(['Condition', 'GroupID', 'AnimalID', 'frame'])
embed_dict = []
for animal_index, animal_id in enumerate(tqdm(np.sort(df.AnimalID.unique()))):
    for condition_index, condition in enumerate(tqdm(np.append(np.sort(df.Condition.unique()),'All'))):
        if condition=='All':
            embed_vals = hfc.loc[(slice(None), slice(None), animal_id, slice(None)), 'embed_vals']
        else:
            embed_vals = hfc.loc[(condition, slice(None), animal_id, slice(None)), 'embed_vals']

        ####################################
        ## Old code does not account for padding
        # n_bins = 1000
        # hist, xedges, yedges = np.histogram2d(
        #     [embed_vals.iloc[i][0] for i in range(np.size(embed_vals))],
        #     [embed_vals.iloc[i][1] for i in range(np.size(embed_vals))],
        #     bins=[n_bins, n_bins],
        #     density=False,
        # )
        ####################################

        ####################################
        # New code for plotting which includes padding
        data = np.array(embed_vals.values.tolist())
        pad_factor = 0.05
        # hist_range = None
        # use hist_range from data_obj
        hist_range = data_obj.ws.hist_range
        n_bins =1000

        range_len = (
            np.ceil(np.amax(data, axis=0)) - np.floor(np.amin(data, axis=0))
        ).astype(int)
        padding = range_len * pad_factor

        # Calculate x and y limits for histogram and density
        if hist_range is None:
            print("Calculating new histogram ranges")
            hist_range = [
                [np.amin(data[:, 0]) - padding[0], np.amax(data[:, 0]) + padding[0]],
                [np.amin(data[:, 1]) - padding[1], np.amax(data[:, 1]) + padding[1]],
            ]
            # self.hist_range = [[int(np.floor(np.amin(data[:,0]))-padding[0]),int(np.ceil(np.amax(data[:,0]))+padding[0])],
            #                    [int(np.floor(np.amin(data[:,1]))-padding[1]),int(np.ceil(np.amax(data[:,1]))+padding[1])]]

        hist, xedges, yedges = np.histogram2d(
            data[:, 0],
            data[:, 1],
            bins=[n_bins, n_bins],
            range=hist_range,
            density=False,
        )
        ####################################
        hist = np.rot90(hist)
        density = hist/np.sum(hist)

        #######################################################
        # Not necessary anymore, calculate when plotting only
        # Should create another table for pairwair comparisons
        # pairedZdensity = np.zeros(np.shape(density))
        # Zdensity = np.zeros(np.shape(density))
        # for cluster_index, cluster_id in enumerate(tqdm(np.sort(df.Cluster.unique()))):
            
        #     # old way does not account for std=0
        #     # pairedZdensity[clusterID_watershed_map==cluster_id] = np.divide(density[clusterID_watershed_map==cluster_id], stdZpaired.loc[(cluster_id, condition, group_id),'frequency'])
        #     #  Zdensity[clusterID_watershed_map==cluster_id] = np.divide(density[clusterID_watershed_map==cluster_id], stdZ.loc[(cluster_id, condition, group_id),'frequency'])
        #     stdZpaired_cluster = stdZpaired.loc[(cluster_id, condition, group_id),'frequency']
        #     if stdZpaired_cluster == 0:
        #         stdZpaired_cluster = 1
        #     pairedZdensity[clusterID_watershed_map==cluster_id] = np.divide(density[clusterID_watershed_map==cluster_id], stdZpaired_cluster)

        #     stdZ_cluster = stdZ.loc[(cluster_id, condition, group_id),'frequency']
        #     if stdZ_cluster == 0:
        #         stdZ_cluster = 1
        #     Zdensity[clusterID_watershed_map==cluster_id] = np.divide(density[clusterID_watershed_map==cluster_id], stdZ_cluster)\
        #######################################################
        density = hist/np.sum(hist) # desnity vairable might be edited by reference?
        # embed_dict.append({'Condition': condition, 'GroupID': group_id, 'density': density, 'pairedZdensity': pairedZdensity, 'counts': hist, 'Zdensity': Zdensity})
        embed_dict.append({'Condition': condition, 'AnimalID': animal_id, 'density': density, 'counts': hist})

hac = pd.DataFrame.from_dict(embed_dict).set_index(['Condition', 'AnimalID'])
hac.to_csv("".join([paths["out_path"], params["label"], "_", parameter_suffix, "/", figure_folder_name, "/hac.csv"]))


############################################
# Calculate tSNE for PD and WT by GroupID from embed_values with Cluster as a predictor
############################################
hfc = df.loc[:, ['AnimalID', 'Condition', 'GroupID', 'frame', 'embed_vals', 'Cluster']].set_index(['Condition', 'GroupID', 'AnimalID', 'frame'])
embed_dict = []
for group_index, group_id in enumerate(tqdm(np.sort(df.GroupID.unique()))):
    for condition_index, condition in enumerate(tqdm(np.append(np.sort(df.Condition.unique()),'All'))):
        if condition=='All':
            embed_vals = hfc.loc[(slice(None), group_id, slice(None), slice(None)), 'embed_vals']
        else:
            embed_vals = hfc.loc[(condition, group_id, slice(None), slice(None)), 'embed_vals']

        ####################################
        ## Old code does not account for padding
        # n_bins = 1000
        # hist, xedges, yedges = np.histogram2d(
        #     [embed_vals.iloc[i][0] for i in range(np.size(embed_vals))],
        #     [embed_vals.iloc[i][1] for i in range(np.size(embed_vals))],
        #     bins=[n_bins, n_bins],
        #     density=False,
        # )
        ####################################

        ####################################
        # New code for plotting which includes padding
        data = np.array(embed_vals.values.tolist())
        pad_factor = 0.05
        # hist_range = None
        # use hist_range from data_obj
        hist_range = data_obj.ws.hist_range
        n_bins =1000

        range_len = (
            np.ceil(np.amax(data, axis=0)) - np.floor(np.amin(data, axis=0))
        ).astype(int)
        padding = range_len * pad_factor

        # Calculate x and y limits for histogram and density
        if hist_range is None:
            print("Calculating new histogram ranges")
            hist_range = [
                [np.amin(data[:, 0]) - padding[0], np.amax(data[:, 0]) + padding[0]],
                [np.amin(data[:, 1]) - padding[1], np.amax(data[:, 1]) + padding[1]],
            ]
            # self.hist_range = [[int(np.floor(np.amin(data[:,0]))-padding[0]),int(np.ceil(np.amax(data[:,0]))+padding[0])],
            #                    [int(np.floor(np.amin(data[:,1]))-padding[1]),int(np.ceil(np.amax(data[:,1]))+padding[1])]]

        hist, xedges, yedges = np.histogram2d(
            data[:, 0],
            data[:, 1],
            bins=[n_bins, n_bins],
            range=hist_range,
            density=False,
        )
        ####################################
        hist = np.rot90(hist)
        density = hist/np.sum(hist)

        #######################################################
        # Not necessary anymore, calculate when plotting only
        # Should create another table for pairwair comparisons
        # pairedZdensity = np.zeros(np.shape(density))
        # Zdensity = np.zeros(np.shape(density))
        # for cluster_index, cluster_id in enumerate(tqdm(np.sort(df.Cluster.unique()))):
            
        #     # old way does not account for std=0
        #     # pairedZdensity[clusterID_watershed_map==cluster_id] = np.divide(density[clusterID_watershed_map==cluster_id], stdZpaired.loc[(cluster_id, condition, group_id),'frequency'])
        #     #  Zdensity[clusterID_watershed_map==cluster_id] = np.divide(density[clusterID_watershed_map==cluster_id], stdZ.loc[(cluster_id, condition, group_id),'frequency'])
        #     stdZpaired_cluster = stdZpaired.loc[(cluster_id, condition, group_id),'frequency']
        #     if stdZpaired_cluster == 0:
        #         stdZpaired_cluster = 1
        #     pairedZdensity[clusterID_watershed_map==cluster_id] = np.divide(density[clusterID_watershed_map==cluster_id], stdZpaired_cluster)

        #     stdZ_cluster = stdZ.loc[(cluster_id, condition, group_id),'frequency']
        #     if stdZ_cluster == 0:
        #         stdZ_cluster = 1
        #     Zdensity[clusterID_watershed_map==cluster_id] = np.divide(density[clusterID_watershed_map==cluster_id], stdZ_cluster)\
        #######################################################
        density = hist/np.sum(hist) # desnity vairable might be edited by reference?
        # embed_dict.append({'Condition': condition, 'GroupID': group_id, 'density': density, 'pairedZdensity': pairedZdensity, 'counts': hist, 'Zdensity': Zdensity})
        embed_dict.append({'Condition': condition, 'GroupID': group_id, 'density': density, 'counts': hist})

hgc = pd.DataFrame.from_dict(embed_dict).set_index(['Condition', 'GroupID'])
hgc.to_csv("".join([paths["out_path"], params["label"], "_", parameter_suffix, "/", figure_folder_name, "/hgc.csv"]))

# Calculate pairwise difference
clusterID_watershed_map =  data_obj.ws.watershed_map
embed_dict = []
for group_index1, group_id1 in enumerate(tqdm(np.sort(df.GroupID.unique()))):
    for group_index2, group_id2 in enumerate(tqdm(np.sort(df.GroupID.unique()))):
        for condition_index, condition in enumerate(tqdm(np.append(np.sort(df.Condition.unique()),'All'))):
            for column_index, column_name in enumerate(tqdm(hgc.columns)):
                       
                metric1 = hgc.loc[(condition, group_id1), column_name]
                metric2 = hgc.loc[(condition, group_id2), column_name]
                std_sum = np.zeros(np.shape(hgc.loc[(condition, group_id1), column_name]))
                for cluster_index, cluster_id in enumerate(tqdm(np.sort(df.Cluster.unique()))):
                    std1 = stdZ.loc[(cluster_id, condition, group_id1),column_name]
                    if std1 == 0:
                            std1 = 1
                    std2 = stdZ.loc[(cluster_id, condition, group_id2),column_name]
                    if std2 == 0:
                            std2 = 1  
                    std_sum[clusterID_watershed_map==cluster_id] = np.sqrt(np.square(std1) + np.square(std2))                

                std_sum[std_sum==0] = 1
                metric_diff = metric1 - metric2
          
                z_score = np.divide(metric_diff, std_sum)

                if column_name == "density":
                    density = metric_diff
                    densityZ = z_score
                if column_name == "counts":
                    counts = metric_diff
                    countsZ = z_score

            embed_dict.append({'Condition': condition, 'GroupID1': group_id1, 'GroupID2': group_id2, 'density': density, 'counts': counts, 'densityZ': densityZ, 'countsZ': countsZ})
hgc_diff = pd.DataFrame.from_dict(embed_dict).set_index(['Condition', 'GroupID1', 'GroupID2'])
hgc_diff.to_csv("".join([paths["out_path"], params["label"],"_", parameter_suffix, "/", figure_folder_name, "/hgc_diff.csv"]))


############################################
# Plot tSNE for PD and WT by AnimalID
############################################
watershed = data_obj.ws
sigma=15
cmap_scheme = 'gist_heat'
cmin = 0
cmax = 3e-6
for animal_index, Animal_id in enumerate(tqdm(np.sort(df.AnimalID.unique()))):
    for condition_index, condition in enumerate(tqdm(np.sort(df.Condition.unique()))):
        label = "AnimalID " + str(animal_id)
        
        plot_density = gaussian_filter(hac.loc[(condition, animal_id),'density'], sigma=sigma) 

        # range_len = (
        #     np.ceil(np.amax(plot_density, axis=0)) - np.floor(np.amin(plot_density, axis=0))
        # ).astype(int)
        # padding = range_len * self.pad_factor

        fig, ax = plt.subplots()
        # sns.color_palette("rocket", as_cmap=True)
        c = sns.heatmap(plot_density, vmin=cmin, vmax=cmax, ax=ax, square=True, cmap=cmap_scheme)
        ax.plot(
            watershed.borders[:, 0],
            watershed.borders[:, 1],
            ".k",
            markersize=0.1,
        )
        ax.set_aspect(0.9)
        ax.set_title(label)
        ax.set_xlabel("t-SNE 1")
        ax.set_ylabel("t-SNE 2")
        ax.set_xticks([])
        ax.set_yticks([])
        fig.tight_layout()
        # f.colorbar(c, ax=ax_arr.ravel().tolist()) # need to set cbar position after calling tight_layout()
        # filepath = "/hpc/group/tdunn/sxl/dappy/results/MCPb_20240206_1GroupIDOnly/fitsne" + parameter_suffix + "/occupancy/Aggregate"
        save_path = "".join([paths["out_path"], params["label"], "_", parameter_suffix, "/", figure_folder_name, "/ByAnimal/Aggregate/" + str(condition)])
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        plt.savefig(save_path + "/" + "Density_Aggregate_GaussFilter-" + str(sigma) + "_ByGroupID" + str(group_id2) + "_cmap-" + cmap_scheme + "-" + str(cmin) + "-" + str(cmax) + ".png", dpi=200)
        plt.close()


############################################
# Plot tSNE for PD and WT by GroupID
############################################
watershed = data_obj.ws
sigma=15
cmap_scheme = 'gist_heat'
cmin = 0
cmax = 3e-6
for group_index2, group_id2 in enumerate(tqdm(np.sort(df.GroupID.unique()))):
    for condition_index, condition in enumerate(tqdm(np.sort(df.Condition.unique()))):
        label = "GroupID " + str(group_id2)
        
        plot_density = gaussian_filter(hgc.loc[(condition, group_id2),'density'], sigma=sigma) 

        # range_len = (
        #     np.ceil(np.amax(plot_density, axis=0)) - np.floor(np.amin(plot_density, axis=0))
        # ).astype(int)
        # padding = range_len * self.pad_factor

        fig, ax = plt.subplots()
        # sns.color_palette("rocket", as_cmap=True)
        c = sns.heatmap(plot_density, vmin=cmin, vmax=cmax, ax=ax, square=True, cmap=cmap_scheme)
        ax.plot(
            watershed.borders[:, 0],
            watershed.borders[:, 1],
            ".k",
            markersize=0.1,
        )
        ax.set_aspect(0.9)
        ax.set_title(label)
        ax.set_xlabel("t-SNE 1")
        ax.set_ylabel("t-SNE 2")
        ax.set_xticks([])
        ax.set_yticks([])
        fig.tight_layout()
        # f.colorbar(c, ax=ax_arr.ravel().tolist()) # need to set cbar position after calling tight_layout()
        # filepath = "/hpc/group/tdunn/sxl/dappy/results/MCPb_20240206_1GroupIDOnly/fitsne" + parameter_suffix + "/occupancy/Aggregate"
        save_path = "".join([paths["out_path"], params["label"], "_", parameter_suffix, "/", figure_folder_name, "/ByGroupID/Aggregate/" + str(condition)])
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        plt.savefig(save_path + "/" + "Density_Aggregate_GaussFilter-" + str(sigma) + "_ByGroupID" + str(group_id2) + "_cmap-" + cmap_scheme + "-" + str(cmin) + "-" + str(cmax) + ".png", dpi=200)
        plt.close()

############################################
# Plot difference in tSNE map between PD and WT by GroupID
############################################
watershed = data_obj.ws
sigma=15
cmap_scheme = 'coolwarm'
cmin = -2e-6
cmax = 2e-6
for group_index1, group_id1 in enumerate(tqdm(np.sort(df.GroupID.unique()))):
    for group_index2, group_id2 in enumerate(tqdm(np.sort(df.GroupID.unique()))):
        for condition_index, condition in enumerate(tqdm(np.sort(df.Condition.unique()))):
            label = "GroupID " + str(group_id1) + "-" + str(group_id2)
            
            # plot_density = gaussian_filter(hgc.loc[(condition, group_id1),'density']-hgc.loc[(condition, group_id2),'density'], sigma=sigma) 
            plot_density = gaussian_filter(hgc_diff.loc[(condition, group_id1, group_id2),'density'], sigma=sigma) 

            # range_len = (
            #     np.ceil(np.amax(plot_density, axis=0)) - np.floor(np.amin(plot_density, axis=0))
            # ).astype(int)
            # padding = range_len * self.pad_factor

            fig, ax = plt.subplots()
            # c = sns.heatmap(plot_density, vmin=cmin, vmax=cmax, ax=ax, square=True, cmap='RdBu')
            c = sns.heatmap(plot_density, vmin=cmin, vmax=cmax, ax=ax, square=True, cmap=cmap_scheme)
            ax.plot(
                watershed.borders[:, 0],
                watershed.borders[:, 1],
                ".k",
                markersize=0.1,
            )
            ax.set_aspect(0.9)
            ax.set_title(label)
            ax.set_xlabel("t-SNE 1")
            ax.set_ylabel("t-SNE 2")
            ax.set_xticks([])
            ax.set_yticks([])
            fig.tight_layout()
            # f.colorbar(c, ax=ax_arr.ravel().tolist()) # need to set cbar position after calling tight_layout()
            # filepath = "/hpc/group/tdunn/sxl/dappy/results/calakos_callie_vps35_cohort2withGrouping/fitsne" + parameter_suffix + "/occupancy/Difference"
            # if not os.path.exists(filepath):
            #     os.makedirs(filepath)  
            # if not os.path.exists(filepath + "/Baseline"):
            #     os.makedirs(filepath + "/Baseline")           
            save_path = "".join([paths["out_path"], params["label"], "_", parameter_suffix, "/", figure_folder_name, "/ByGroupID/Difference/" + str(condition)])
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            plt.savefig(save_path + "/" + "Density_Difference_GaussFilter-" + str(sigma) + "_ByGroupID" + str(group_id1) + "-"  + str(group_id2) + "_cmap-" + cmap_scheme + "-" + str(cmin) + "-" + str(cmax) + ".png", dpi=200)
            plt.close()




# ###########################################
# # Plot density normalized by std dev (not mathematically accurate, need to subtract off mean)
# ###########################################
# watershed = data_obj.ws
# sigma=15
# cmap_scheme = 'coolwarm'
# cmin = 0
# cmax = 1.5e-3

# for group_index2, group_id2 in enumerate(np.sort(df.GroupID.unique())):
#     # for condition_index, condition in enumerate(np.sort(df.Condition.unique())):
#     for condition in ["Baseline"]:
#         label = "GroupID " + str(group_id2)
        
#         # plot_density = gaussian_filter(hgc.loc[(condition, group_id2),'Zdensity'], sigma=sigma)  

#         density2 = hgc.loc[(condition, group_id2),'density']
#         std2 = stdZ.loc[(cluster_id, condition, group_id2),'frequency']
#         if std2 == 0:
#                 std2 = 1
#         plot_density = gaussian_filter(np.divide(density2,std2), sigma=sigma)  

#         # range_len = (
#         #     np.ceil(np.amax(plot_density, axis=0)) - np.floor(np.amin(plot_density, axis=0))
#         # ).astype(int)
#         # padding = range_len * self.pad_factor

#         fig, ax = plt.subplots()
#         # sns.color_palette("rocket", as_cmap=True)
#         c = sns.heatmap(plot_density, vmin=cmin, vmax=cmax, ax=ax, square=True, cmap='gist_heat')
#         ax.plot(
#             watershed.borders[:, 0],
#             watershed.borders[:, 1],
#             ".k",
#             markersize=0.1,
#         )
#         ax.set_aspect(0.9)
#         ax.set_title(label)
#         ax.set_xlabel("t-SNE 1")
#         ax.set_ylabel("t-SNE 2")
#         ax.set_xticks([])
#         ax.set_yticks([])
#         fig.tight_layout()
#         # f.colorbar(c, ax=ax_arr.ravel().tolist()) # need to set cbar position after calling tight_layout()
#         # filepath = "/hpc/group/tdunn/sxl/dappy/results/MCPb_20240206_1GroupIDOnly/fitsne" + parameter_suffix + "/occupancy/Aggregate"
#     filepath = "".join([paths["out_path"], params["label"],parameter_suffix]) + "/" + figure_folder_name + "/Aggregate_DivideByStd/"   + str(condition) + "/"  
#     if not os.path.exists(filepath):
#         os.makedirs(filepath)
#     # plt.savefig(filepath + "occupancy_diff_Group" + str(group_id1) + "-" +str(group_id2), dpi=200)
#     plt.savefig("".join([filepath, "Density_Aggregate_DivByStd_GaussFilter-" + str(sigma) + "_ByGroupID" + str(group_id2) + "_cmap-" + cmap_scheme + "-" + str(cmin) + "-" + str(cmax)+ ".png"]), dpi=400)
#     plt.close()

########################
# Pairwise difference normalized by combined paired std (density divided by standard deviation calculated from frequency)
########################
ws = data_obj.ws
sigma=15
cmap_scheme = 'coolwarm'
cmin = -0.00025
cmax = 0.00025
# condition = "Baseline"

for condition_index, condition in enumerate(tqdm(np.append(np.sort(df.Condition.unique()),'All'))):
    for group_index1, group_id1 in enumerate(tqdm(np.sort(df.GroupID.unique()))):
        for group_index2, group_id2 in enumerate(tqdm(np.sort(df.GroupID.unique()))):    
            label = "Group " + str(group_id1) + " - Group " + str(group_id2)

            # plot_density = gaussian_filter(hgc.loc[(condition, group_id1),'pairedZdensity']-hgc.loc[(condition, group_id2),'pairedZdensity'], sigma=sigma) 
            
            # density1 = hgc.loc[(condition, group_id1),'density']
            # density2 = hgc.loc[(condition, group_id2),'density']
            # std1 = stdZ.loc[(cluster_id, condition, group_id1),'frequency']
            # if std1 == 0:
            #         std1 = 1
            # std2 = stdZ.loc[(cluster_id, condition, group_id2),'frequency']
            # if std2 == 0:
            #         std2 = 1    
            # plot_density = gaussian_filter(np.divide(density1-density2,np.sqrt(np.square(std1)+np.square(std2))), sigma=sigma)

            plot_density = gaussian_filter(hgc_diff.loc[(condition, group_id1, group_id2),'densityZ'], sigma=sigma) 

            # range_len = (
            #     np.ceil(np.amax(plot_density, axis=0)) - np.floor(np.amin(plot_density, axis=0))
            # ).astype(int)
            # padding = range_len * self.pad_factor

            fig, ax = plt.subplots()
            # c = sns.heatmap(plot_density, vmin=cmin, vmax=cmax, ax=ax, square=True, cmap='RdBu')
            c = sns.heatmap(plot_density, vmin=cmin, vmax=cmax, ax=ax, square=True, cmap=cmap_scheme)
            ax.plot(
                ws.borders[:, 0],
                ws.borders[:, 1],
                ".k",
                markersize=0.1,
            )
            ax.set_aspect(0.9)
            ax.set_title(label)
            ax.set_xlabel("t-SNE 1")
            ax.set_ylabel("t-SNE 2")
            ax.set_xticks([])
            ax.set_yticks([])
            fig.tight_layout()
            # f.colorbar(c, ax=ax_arr.ravel().tolist()) # need to set cbar position after calling tight_layout()
            filepath = "".join([paths["out_path"], params["label"], "_", parameter_suffix]) + "/" + figure_folder_name + "/ByGroupID/Difference_DivByPairedStd/"   + str(condition) + "/"  
            if not os.path.exists(filepath):
                os.makedirs(filepath)
            # plt.savefig(filepath + "occupancy_diff_Group" + str(group_id1) + "-" +str(group_id2), dpi=200)
            plt.savefig("".join([filepath, "Density_Difference_DivByPairedStd_GaussFilter-" + str(sigma) + "_BetweenGroupID" + str(group_id1) + "-" + str(group_id2) + "_cmap-" + cmap_scheme + "-" + str(cmin) + "-" + str(cmax)+ ".png"]), dpi=400)
            plt.close()

########################################
# Take sum of density across clusters
# This is the correct approach, taking mean of density within cluster skews value based on number of points in cluster
########################################
# ws = data_obj.ws
# meanDensity_dict = []
# for group_index, group_id in enumerate(np.sort(df.GroupID.unique())):
#     for condition_index, condition in enumerate(np.append(np.sort(df.Condition.unique()),'All')):
        
#         density = hgc.loc[(condition, group_id), 'density']
#         # pairedZdensity = hgc.loc[(condition, group_id), 'pairedZdensity']
#         counts = hgc.loc[(condition, group_id), 'counts']

#         group_map2 = np.zeros(np.shape(hgc.loc[(condition, group_id), 'density']))
#         # group_map = np.zeros(np.shape(hgc.loc[(condition, group_id), 'pairedZdensity']))
#         group_map3 = np.zeros(np.shape(hgc.loc[(condition, group_id), 'counts']))
#         for cluster_index, cluster_id in enumerate(tqdm(np.sort(df.Cluster.unique()))):
#             # group_map[ws.watershed_map==cluster_id] = np.sum(pairedZdensity[ws.watershed_map==cluster_id])
#             group_map2[ws.watershed_map==cluster_id] = np.sum(density[ws.watershed_map==cluster_id])
#             group_map3[ws.watershed_map==cluster_id] = np.sum(counts[ws.watershed_map==cluster_id])               

#         # meanDensity_dict.append({'Condition': condition, 'GroupID': group_id, 'density': density, 'pairedZdensity': pairedZdensity, 'sum_density': group_map2, 'sum_pairedZdensity': group_map, 'sum_counts': group_map3})
#         meanDensity_dict.append({'Condition': condition, 'GroupID': group_id, 'density': density, 'sum_density': group_map2, 'sum_counts': group_map3})

# hgc_sum = pd.DataFrame.from_dict(meanDensity_dict).set_index(['Condition', 'GroupID'])


# ws = data_obj.ws
sumDensity_dict = []
for group_index, group_id in enumerate(np.sort(df.GroupID.unique())):
    for condition_index, condition in enumerate(np.append(np.sort(df.Condition.unique()),'All')):
        for column_index, column_name in enumerate(hgc.columns):
            metric =  hgc.loc[(condition, group_id), column_name]
            group_map = np.zeros(np.shape(hgc.loc[(condition, group_id), column_name]))

            for cluster_index, cluster_id in enumerate(tqdm(np.sort(df.Cluster.unique()))):
                group_map[clusterID_watershed_map==cluster_id] = np.sum(metric[clusterID_watershed_map==cluster_id])

            if column_name == "density":
                density = group_map
            if column_name == "counts":
                counts = group_map              

        sumDensity_dict.append({'Condition': condition, 'GroupID': group_id, 'density': density, 'counts': counts})

sgc = pd.DataFrame.from_dict(sumDensity_dict).set_index(['Condition', 'GroupID'])
sgc.to_csv("".join([paths["out_path"], params["label"], "_", parameter_suffix, "/", figure_folder_name, "/sgc" + ".csv"]))

sumDensity_dict = []
for group_index1, group_id1 in enumerate(tqdm(np.sort(df.GroupID.unique()))):
    for group_index2, group_id2 in enumerate(tqdm(np.sort(df.GroupID.unique()))):
        for condition_index, condition in enumerate(tqdm(np.append(np.sort(df.Condition.unique()),'All'))):
            for column_index, column_name in enumerate(tqdm(hgc.columns)):
                    
                metric1 = sgc.loc[(condition, group_id1), column_name]
                metric2 = sgc.loc[(condition, group_id2), column_name]
                std_sum = np.zeros(np.shape(sgc.loc[(condition, group_id1), column_name]))
                for cluster_index, cluster_id in enumerate(tqdm(np.sort(df.Cluster.unique()))):
                    std1 = stdZ.loc[(cluster_id, condition, group_id1),column_name]
                    if std1 == 0:
                            std1 = 1
                    std2 = stdZ.loc[(cluster_id, condition, group_id2),column_name]
                    if std2 == 0:
                            std2 = 1  
                    std_sum[clusterID_watershed_map==cluster_id] = np.sqrt(np.square(std1) + np.square(std2))
                    
                std_sum[std_sum==0] = 1
                metric_diff = metric1 - metric2             
                z_score = np.divide(metric_diff, std_sum)

                if column_name == "density":
                    density = metric_diff
                    densityZ = z_score
                if column_name == "counts":
                    counts = metric_diff
                    countsZ = z_score

            sumDensity_dict.append({'Condition': condition, 'GroupID1': group_id1, 'GroupID2': group_id2, 'density': density, 'counts': counts, 'densityZ': densityZ, 'countsZ': countsZ})
sgc_diff = pd.DataFrame.from_dict(sumDensity_dict).set_index(['Condition', 'GroupID1', 'GroupID2'])
sgc_diff.to_csv("".join([paths["out_path"], params["label"], "_", parameter_suffix, "/", figure_folder_name, "/sgc_diff.csv"]))



########################
# test pairwise comparison plot (sum of density within clusters BEFORE dividing by std dev)
########################
# ws = data_obj.ws
watershed = data_obj.ws
sigma=15
cmap_scheme = 'coolwarm'
cmin = -0.012
cmax = 0.012
# condition = "Baseline"

for condition_index, condition in enumerate(tqdm(np.append(np.sort(df.Condition.unique()),'All'))):
    for group_index1, group_id1 in enumerate(tqdm(np.sort(df.GroupID.unique()))):
        for group_index2, group_id2 in enumerate(tqdm(np.sort(df.GroupID.unique()))):
            label = "Group " + str(group_id1) + " - Group " + str(group_id2)

            # plot_density = gaussian_filter(hgc_mean.loc[(condition, group_id1),'mean_pairedZdensity']-hgc_mean.loc[(condition, group_id2),'mean_pairedZdensity'], sigma=sigma) 
            # plot_density = sgc.loc[(condition, group_id1),'density']-sgc.loc[(condition, group_id2),'density']
            plot_density = sgc_diff.loc[(condition, group_id1, group_id2),'density']

            # # See effects of adding filter, looks nice but not accurate
            # plot_density = gaussian_filter(plot_density, sigma=sigma)

            # range_len = (
            #     np.ceil(np.amax(plot_density, axis=0)) - np.floor(np.amin(plot_density, axis=0))
            # ).astype(int)
            # padding = range_len * self.pad_factor

            fig, ax = plt.subplots()
            # c = sns.heatmap(plot_density, vmin=cmin, vmax=cmax, ax=ax, square=True, cmap='RdBu')
            c = sns.heatmap(plot_density, vmin=cmin, vmax=cmax, ax=ax, square=True, cmap=cmap_scheme)
            ax.plot(
                watershed.borders[:, 0],
                watershed.borders[:, 1],
                ".k",
                markersize=0.1,
            )
            ax.set_aspect(0.9)
            ax.set_title(label)
            ax.set_xlabel("t-SNE 1")
            ax.set_ylabel("t-SNE 2")
            ax.set_xticks([])
            ax.set_yticks([])
            fig.tight_layout()
            # f.colorbar(c, ax=ax_arr.ravel().tolist()) # need to set cbar position after calling tight_layout()
            filepath = "".join([paths["out_path"], params["label"],"_", parameter_suffix]) + "/" + figure_folder_name + "/ByGroupID/DifferenceOfSum/"   + str(condition) + "/"  
            if not os.path.exists(filepath):
                os.makedirs(filepath)
            # plt.savefig(filepath + "occupancy_diff_Group" + str(group_id1) + "-" +str(group_id2), dpi=200)
            plt.savefig("".join([filepath, "sumDensity_Difference_noGaussFilter_" + "BetweenGroupID" + str(group_id1) + "-" + str(group_id2) + "_cmap-" + cmap_scheme + "-" + str(cmin) + "-" + str(cmax)+ ".png"]), dpi=400)
            plt.close()             

########################
# test pairwise comparison plot (sum of density within clusters divided by std dev)
########################
# ws = data_obj.ws
watershed = data_obj.ws
sigma=15
cmap_scheme = 'coolwarm'
cmin = -1.2
cmax = 1.2
# condition = "Baseline"

for condition_index, condition in enumerate(tqdm(np.append(np.sort(df.Condition.unique()),'All'))):
    for group_index1, group_id1 in enumerate(tqdm(np.sort(df.GroupID.unique()))):
        for group_index2, group_id2 in enumerate(tqdm(np.sort(df.GroupID.unique()))):
            label = "Group " + str(group_id1) + " - Group " + str(group_id2)

            # plot_density = gaussian_filter(hgc_mean.loc[(condition, group_id1),'mean_pairedZdensity']-hgc_mean.loc[(condition, group_id2),'mean_pairedZdensity'], sigma=sigma) 
            # plot_density = hgc_mean.loc[(condition, group_id1),'mean_density']-hgc_mean.loc[(condition, group_id2),'mean_density']

            # density1 = sgc.loc[(condition, group_id1),'sum_density']
            # density2 = sgc.loc[(condition, group_id2),'sum_density']
            # std1 = stdZ.loc[(cluster_id, condition, group_id1),'frequency']
            # if std1 == 0:
            #         std1 = 1
            # std2 = stdZ.loc[(cluster_id, condition, group_id2),'frequency']
            # if std2 == 0:
            #         std2 = 1                
            
            
            # # plot_density = gaussian_filter(np.divide(density1-density2,np.sqrt(np.square(std1)+np.square(std2))), sigma=sigma)
            # plot_density = np.divide(density1-density2,np.sqrt(np.square(std1)+np.square(std2)))

            plot_density = sgc_diff.loc[(condition, group_id1, group_id2),'densityZ']


            # # See effects of adding filter, looks nice but not accurate
            # plot_density = gaussian_filter(plot_density, sigma=sigma)

            # range_len = (
            #     np.ceil(np.amax(plot_density, axis=0)) - np.floor(np.amin(plot_density, axis=0))
            # ).astype(int)
            # padding = range_len * self.pad_factor

            fig, ax = plt.subplots()
            # c = sns.heatmap(plot_density, vmin=cmin, vmax=cmax, ax=ax, square=True, cmap='RdBu')
            c = sns.heatmap(plot_density, vmin=cmin, vmax=cmax, ax=ax, square=True, cmap=cmap_scheme)
            ax.plot(
                watershed.borders[:, 0],
                watershed.borders[:, 1],
                ".k",
                markersize=0.1,
            )
            ax.set_aspect(0.9)
            ax.set_title(label)
            ax.set_xlabel("t-SNE 1")
            ax.set_ylabel("t-SNE 2")
            ax.set_xticks([])
            ax.set_yticks([])
            fig.tight_layout()
            # f.colorbar(c, ax=ax_arr.ravel().tolist()) # need to set cbar position after calling tight_layout()
            filepath = "".join([paths["out_path"], params["label"],"_", parameter_suffix]) + "/" + figure_folder_name + "/ByGroupID/DifferenceOfSum_DivByPairedStd/"   + str(condition) + "/"  
            if not os.path.exists(filepath):
                os.makedirs(filepath)
            # plt.savefig(filepath + "occupancy_diff_Group" + str(group_id1) + "-" +str(group_id2), dpi=200)
            plt.savefig("".join([filepath, "sumDensity_Difference_DivByPairedStd_noGaussFilter_" + "BetweenGroupID" + str(group_id1) + "-" + str(group_id2) + "_cmap-" + cmap_scheme + "-" + str(cmin) + "-" + str(cmax)+ ".png"]), dpi=400)
            plt.close()     




##################################
# Add SXL 20240713, Find difference between sessions
##################################
watershed = data_obj.ws
sigma=15
cmin = -2e-6
cmax = 2e-6
for condition_index1, condition1 in enumerate(tqdm(np.append(np.sort(df.Condition.unique()),'All'))):
    for condition_index2, condition2 in enumerate(tqdm(np.append(np.sort(df.Condition.unique()),'All'))):
        for group_index, group_id in enumerate(tqdm(np.sort(df.GroupID.unique()))):
        
        # for condition_index, condition in enumerate(["Baseline"]):
            label = str(condition1) + " - " + str(condition2)
            
            plot_density = gaussian_filter(hg.loc[(condition1, group_id),'density']-hg.loc[(condition2, group_id),'density'], sigma=sigma) 

            # range_len = (
            #     np.ceil(np.amax(plot_density, axis=0)) - np.floor(np.amin(plot_density, axis=0))
            # ).astype(int)
            # padding = range_len * self.pad_factor
   
            fig, ax = plt.subplots()
            # c = sns.heatmap(plot_density, vmin=cmin, vmax=cmax, ax=ax, square=True, cmap='RdBu')
            c = sns.heatmap(plot_density, vmin=cmin, vmax=cmax, ax=ax, square=True, cmap='coolwarm')
            ax.plot(
                watershed.borders[:, 0],
                watershed.borders[:, 1],
                ".k",
                markersize=0.1,
            )
            ax.set_aspect(0.9)
            ax.set_title(label)
            ax.set_xlabel("t-SNE 1")
            ax.set_ylabel("t-SNE 2")
            ax.set_xticks([])
            ax.set_yticks([])
            fig.tight_layout()
            # f.colorbar(c, ax=ax_arr.ravel().tolist()) # need to set cbar position after calling tight_layout()
            filepath = paths["out_path"] + params["label"] + "_" + parameter_suffix + "/" + figure_folder_name + "/occupancyWithPadding/Difference"
            if not os.path.exists(filepath + "/" + str(group_id) + "/"):
                os.makedirs(filepath + "/" + str(group_id) + "/")            
            plt.savefig(filepath + "/" + str(group_id) + "/" + "occupancy_diff_Session" + str(condition1) + "-" +str(condition2), dpi=200)
            plt.close()






