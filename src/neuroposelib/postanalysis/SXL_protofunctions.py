# # from features import *
# import DataStruct as ds
# # import visualization as vis
# import numpy as np
# import pandas as pd
# import read, write
# # from embed import Watershed, Embed
# import pickle
# # import analysis


from neuroposelib.postanalysis.features import *
import neuroposelib.DataStruct as ds
import neuroposelib.visualization as vis
import numpy as np
import pandas as pd
from neuroposelib import read, write
from neuroposelib.embed import Watershed, Embed
import pickle
from neuroposelib import analysis

import re
from tqdm import tqdm

import matplotlib.pyplot as plt
# from scipy import stats
from statsmodels import stats 
# needed to install statsmodels via pip install not conda install, could not import when using conda install

from matplotlib.colors import LogNorm
from scipy.ndimage import gaussian_filter

import seaborn as sns




def control_fdr(pvalue, fdr, sorted = False):

    # Benjamini-HOchberg is only good if the mulitple measurements tested are independent or positively correlated
    # If measurements are negatively correlated have to use the BY method
    # https://www.graphpad.com/guides/prism/latest/statistics/stat_pros_and_cons_of_the_three_met.htm


    # DANNCE Clusters should be negatively correlated since it they always have to sum to 1


    n_tests = np.size(pvalue)

    id = np.arange(n_tests, dtype=int)
    rank = np.linspace(1, n_tests, n_tests, dtype=int)
    critical_value = rank/n_tests*fdr

    table = pd.DataFrame({'id': id, 'pvalue': pvalue})
    
    ranked = table.sort_values(by='pvalue', axis='index')

    ranked_pvalue = ranked.loc[:,'pvalue'].values
    ranked['rank'] = rank
    ranked['critical_value'] = critical_value

    # compare p-values to critical values
    ranked['is_lesser'] = ranked.loc[:,'pvalue'] < ranked.loc[:,'critical_value'] 
    lesser_ranks_index = np.where(ranked.is_lesser)

    if np.size(lesser_ranks_index):
        last_lesser_rank_index = np.max(lesser_ranks_index)
        ranked['is_significant'] = ranked.loc[:,'rank'] <= ranked.iloc[last_lesser_rank_index]['rank']        
    else:
        last_lesser_rank_index = []
        ranked['is_significant'] = False
   
    adjusted_pvalue = np.array([  np.min([ np.size(rank) * ranked_pvalue[j-1]/j for j in range( i, np.size(rank)+1 )  ])   for i in rank  ])
    adjusted_pvalue = np.minimum(np.ones(np.size(adjusted_pvalue)), adjusted_pvalue) 

    ranked['adjusted_pvalue'] = adjusted_pvalue                 

    if (sorted):
        ranked.sort_values(by='id',axis=0)

    return ranked



def bootstrap_twosample(first_group, second_group, n_sampling=10000, metric='mean'):
    
    # This method only generates null distribution from the provided groups, but in our dataset we have more data for each cluster among groups

    first_group = np.squeeze(first_group)
    second_group = np.squeeze(second_group)

    first_metric = np.mean(first_group)
    second_metric = np.mean(second_group)
    difference_metric = second_metric - first_metric

    first_size = np.size(first_group)
    second_size = np.size(second_group)
    
    merged = np.concatenate([first_group, second_group], axis=0)
    bootstrap_null = np.empty([n_sampling,])
    print('Bootstrapping null distribution...' + "\n")
    for index_sampling in tqdm(range(n_sampling)):
        first_group_boot = np.random.choice(merged, size=first_size, replace=True, p=None)
        second_group_boot = np.random.choice(merged, size=second_size, replace=True, p=None)
        first_metric_boot = np.mean(first_group_boot)
        second_metric_boot = np.mean(second_group_boot)
        difference_metric_boot = second_metric_boot - first_metric_boot
        bootstrap_null[index_sampling] = difference_metric_boot

    # calculate p-value (two-tailed)
    freq_greater = np.sum(bootstrap_null>=np.absolute(difference_metric))/np.size(bootstrap_null)
    freq_lesser = np.sum(bootstrap_null<=-np.absolute(difference_metric))/np.size(bootstrap_null)
    p_value = freq_greater + freq_lesser

    return difference_metric, p_value, bootstrap_null
    # return p_value


def bootstrap_twosample_covariate(first_group, second_group, n_sampling=10000, metric='mean'):
    
    # This method only generates null distribution from the provided groups, but in our dataset we have more data for each cluster among groups

    first_group = np.squeeze(first_group)
    second_group = np.squeeze(second_group)

    first_metric = np.mean(first_group, axis=0)
    second_metric = np.mean(second_group, axis=0)
    difference_metric = second_metric - first_metric

    first_size = np.shape(first_group)[0]
    second_size = np.shape(second_group)[0]

    if np.shape(first_group)[1] == np.shape(second_group)[1]:
        n_clusters = np.shape(first_group)[1]
    else:
        print("Groups provided have different number of measurements")
        # return 
    
    merged = np.concatenate([first_group, second_group], axis=0)
    bootstrap_null = np.empty([n_sampling,n_clusters])

    rng = np.random.default_rng()

    print('Bootstrapping null distribution...' + "\n")
    for index_sampling in tqdm(range(n_sampling)):
        first_group_boot = rng.choice(merged, size=first_size, replace=True, p=None, axis=0, shuffle=False)
        second_group_boot = rng.choice(merged, size=second_size, replace=True, p=None, axis=0, shuffle=False)
        first_metric_boot = np.mean(first_group_boot, axis=0)
        second_metric_boot = np.mean(second_group_boot, axis=0)
        difference_metric_boot = second_metric_boot - first_metric_boot
        bootstrap_null[index_sampling,:] = difference_metric_boot

    # calculate p-value (two-tailed)
    p_value = np.empty([n_clusters])
    for index_cluster in tqdm(range(n_clusters)):
        # freq_greater = np.sum(bootstrap_null>=np.absolute(difference_metric))/np.shape(bootstrap_null)[0]
        # freq_lesser = np.sum(bootstrap_null<=-np.absolute(difference_metric))/np.shape(bootstrap_null)[0]
        # p_value[index_cluster] = freq_greater + freq_lesser
        
        p_value[index_cluster] = (np.sum(bootstrap_null[:,index_cluster]>=np.absolute(difference_metric.iloc[index_cluster])) + np.sum(bootstrap_null[:,index_cluster]<=-np.absolute(difference_metric.iloc[index_cluster])))/np.shape(bootstrap_null)[0]
    return difference_metric, p_value, bootstrap_null
    # return p_value

    

def bootstrap_twosample_fullnull(first_group, second_group, merged, n_sampling=10000, metric='mean'):
    
    # This method only generates null distribution from the provided groups, but in our dataset we have more data for each cluster among groups

    first_group = np.squeeze(first_group)
    second_group = np.squeeze(second_group)

    first_metric = np.mean(first_group)
    second_metric = np.mean(second_group)
    difference_metric = second_metric - first_metric

    first_size = np.size(first_group)
    second_size = np.size(second_group)
    
    # merged = np.concatenate([first_group, second_group], axis=0)
    bootstrap_null = np.empty([n_sampling,])
    print('Bootstrapping null distribution...' + "\n")
    for index_sampling in tqdm(range(n_sampling)):
        first_group_boot = np.random.choice(merged, size=first_size, replace=True, p=None)
        second_group_boot = np.random.choice(merged, size=second_size, replace=True, p=None)
        first_metric_boot = np.mean(first_group_boot)
        second_metric_boot = np.mean(second_group_boot)
        difference_metric_boot = second_metric_boot - first_metric_boot
        bootstrap_null[index_sampling] = difference_metric_boot

    # calculate p-value (two-tailed)
    freq_greater = np.sum(bootstrap_null>=np.absolute(difference_metric))/np.size(bootstrap_null)
    freq_lesser = np.sum(bootstrap_null<=-np.absolute(difference_metric))/np.size(bootstrap_null)
    p_value = freq_greater + freq_lesser

    return difference_metric, p_value, bootstrap_null
    # return p_value



def bootstrap_twosample_shuffle(first_group, second_group, n_sampling=10000, metric='mean'):

    # bootstrap without replacement
    
    first_group = np.squeeze(first_group)
    second_group = np.squeeze(second_group)

    first_metric = np.mean(first_group)
    second_metric = np.mean(second_group)
    difference_metric = second_metric - first_metric

    first_size = np.size(first_group)
    second_size = np.size(second_group)
    
    merged = np.concatenate([first_group, second_group], axis=0)
    merged_shuffled = merged
    bootstrap_null = np.empty([n_sampling,])
    print('Bootstrapping null distribution...' + "\n")
    for index_sampling in tqdm(range(n_sampling)):
        np.random.shuffle(merged_shuffled)
        first_group_boot = merged_shuffled[0:first_size]
        second_group_boot = merged_shuffled[first_size:first_size+second_size]
        first_metric_boot = np.mean(first_group_boot)
        second_metric_boot = np.mean(second_group_boot)
        difference_metric_boot = second_metric_boot - first_metric_boot
        bootstrap_null[index_sampling] = difference_metric_boot

    # calculate p-value (two-tailed)
    freq_greater = np.sum(bootstrap_null>=np.absolute(difference_metric))/np.size(bootstrap_null)
    freq_lesser = np.sum(bootstrap_null<=-np.absolute(difference_metric))/np.size(bootstrap_null)
    p_value = freq_greater + freq_lesser

    return difference_metric, p_value, bootstrap_null
    # return p_value



def get_transition(state, unique_states, output="absolute"):
    # creates a square trasition matrix, T, where S2=T*S1
    # row represent new state
    # coumn represents old state
    # # i.e. row 2 column 3 is the transition probability from state 3 going to state 2 
    # State is a time series of the states across time

    # state_change = np.diff(state)
    # for index, change in enumerate(state_change):
    for index, post in enumerate(state):
        if (index==0):         
            # unique_states = np.sort(np.unique(state)) # need to deal with missing state numbers in the future, i.e. if cluter 40 was dropped when we have clusters 1 to 90
            # n_states = np.size(unique_states)
            # # state_rows = np.arange(n_states) 
            # min_state_id = min(state)
            # max_state_id = max(state)
            # print("No. of unique states: " + str(n_states))
            # print("Lowest state id: " + str(min_state_id))
            # print("Highest state id: " + str(max_state_id))
            # if len(range(n_states)) != len(range(min_state_id, max_state_id+1)):
            #     print("List of all possible state ids is not continous")
            #     n_size = max(state)
            # else:
            #     n_size = n_states

            n_states = np.size(unique_states)
            n_size = n_states
            transition = np.zeros([n_size, n_size])
        else:
            transition[post-1,pre-1] += 1 # start from 0 for index
        pre=post

    if output == "probability":
        transition = transition/np.sum(transition, axis=None)

    return transition


def density_grid_norm(
    data: ds.DataStruct,
    cat1: str,
    cat2: str,
    watershed: Watershed,
    filepath: str = "./plot_folder/density_by_label.png",
    norm=2.5,
    show: bool = False,
):
    """
    Plot densities by a category label
    """
    labels1, labels2 = data.data[cat1].values, data.data[cat2].values
    n_col = len(np.unique(labels2))
    n_rows = len(np.unique(labels1))
    f, ax_arr = plt.subplots(n_rows, n_col, figsize=((n_col + 1) * 4, n_rows * 4))

    # Loop over unique labels
    for i, label1 in enumerate(np.unique(labels1)):
        ax_arr[i, 0].set_title(label1)
        for j, label2 in enumerate(np.unique(labels2)):
            # import pdb; pdb.set_trace()
            embed_vals = data.embed_vals[
                (data.data[cat1] == label1) & (data.data[cat2] == label2)
            ]  # Indexing by label
            density = watershed.fit_density(
                embed_vals, new=False
            )  # Fit density on old axes
            if n_rows == 1:
                # ax_arr[j].imshow(density)  # scp.special.softmax(density))
                ax_arr[j].imshow(density, vmax=2.5)  # scp.special.softmax(density)) # SXLedit 20230920
                if watershed is not None:
                    ax_arr[j].plot(
                        watershed.borders[:, 0],
                        watershed.borders[:, 1],
                        ".k",
                        markersize=0.1,
                    )
                ax_arr[j].set_aspect("auto")
                ax_arr[j].set_title(label1)
                ax_arr[j].set_xlabel("t-SNE 1")
                ax_arr[j].set_ylabel("t-SNE 2")
                ax_arr[j].set_xticks([])
                ax_arr[j].set_yticks([])
            else:
                ax_arr[i, j].imshow(scp.special.softmax(density))

                if watershed is not None:
                    ax_arr[i, j].plot(
                        watershed.borders[:, 0],
                        watershed.borders[:, 1],
                        ".k",
                        markersize=0.1,
                    )
                if i == 0:
                    ax_arr[0, j].set_title(label2)
                ax_arr[i, j].set_aspect("auto")
                ax_arr[i, j].set_xlabel("t-SNE 1")
                ax_arr[i, j].set_ylabel("t-SNE 2")
                ax_arr[i, j].set_xticks([])
                ax_arr[i, j].set_yticks([])
    f.tight_layout()
    plt.savefig(filepath, dpi=200)
    if show:
        plt.show()
    plt.close()
    return


def density_cat_norm(
    data: ds.DataStruct,
    column: str,
    watershed: Watershed,
    filepath: str = "./plot_folder/density_by_label.png",
    n_col: int = 4,
    transform = 'linear',
    cscale = 1, # used to define scale factor for visualization
    cmin = 0,
    cmax = None,
    show: bool = False,
):
    """
    Plot densities by a category label
    """
    labels = data.data[column].values
    n_col = min(n_col, len(np.unique(labels)))
    n_rows = int(np.ceil(len(np.unique(labels)) / n_col))
    f, ax_arr = plt.subplots(n_rows, n_col, figsize=((n_col + 1) * 4, n_rows * 4))

    # Loop over unique labels
    for i, label in enumerate(np.unique(labels)):
        embed_vals = data.embed_vals[data.data[column] == label]  # Indexing by label
        
        # density = watershed.fit_density(
        #     embed_vals, new=False
        # )  # Fit density on old axes
        
        ###################################
        # manual fit density
                # 2D histogram


        # coded hist functino in watershed seems problematic
        # new = False        
        # hist = watershed.hist(embed_vals, new)
        # density = hist/np.sum(hist)

        
        n_bins = 1000
        hist, xedges, yedges = np.histogram2d(
            embed_vals[:, 0],
            embed_vals[:, 1],
            bins=[n_bins, n_bins],
            # range=self.hist_range,
            density=False,
        )
        hist = np.rot90(hist)

        # density = hist

        density = hist/np.sum(hist)
        

        # density = gaussian_filter(density, sigma=watershed.sigma)

        # Calculates density using gaussian filter
        # density = gaussian_filter(hist, sigma=watershed.sigma)
        # # if self.log_out:
        # #     density = np.log1p(density)
        # density = np.clip(
        #     density, None, np.amax(density) * watershed.max_clip
        # )  # clips max for better visualization of clusters

        # if new:
        #     density = density

        ############################################
     
        col_i = i % n_col
        row_i = int(i/n_col)

        # scale density by number of data points (no. or rows) in category
        n_points = np.shape(embed_vals)[0]
        n_pixels = np.shape(density)[0]*np.shape(density)[1]

        
        
        # set scale to assume each point 

        # plot_density = density
        sigma=15
        plot_density = gaussian_filter(density, sigma=sigma)    
        # without gaussian filter, appearsnce of density seems to be correlated with number of samples  

        # plot_density = density

        # plot_density = density/n_points*n_pixels


        # cmax = n_points/n_pixels * cscale

        if(transform=='linear'):          
            plot_density = plot_density
        elif(transform=='softmax'):
            plot_density = scp.special.softmax(plot_density)
        elif(transform=='log2'):
            plot_density = np.log2(1+plot_density)
        elif(transform=='log2_ignore'):
            plot_density[plot_density!=0] = -np.log2(plot_density[plot_density!=0]) 
            plot_density[plot_density!=0] = plot_density[plot_density!=0]/-np.log2(cmax)     
        elif(transform=='log10'):
            plot_density = np.log10(1+plot_density)
        elif(transform=='log10_ignore'):
            plot_density[plot_density!=0] = -np.log10(plot_density[plot_density!=0]) 
            plot_density[plot_density!=0] = plot_density[plot_density!=0]/-np.log10(cmax)   
        elif(transform=='log10_before'):
            plot_density = np.log10(1+hist)/np.log10(1+cmax)      
        else:
            plot_density = plot_density        


        print("GroupID: " + str(label))
        print("No. of observations: " + str(n_points))
        print("No. of pixels in map: " + str(n_pixels))
        print("Sum of density: " + str(np.sum(plot_density)))
        print("No. of nonzero pixels: " + str(np.sum(plot_density!=0)))
        print("Max density (ignoring zeros): " + str(np.max(plot_density[plot_density!=0])))
        print("Mean density (ignoring zeros): " + str(np.mean(plot_density[plot_density!=0])))
        print("Median density (ignoring zeros): " + str(np.median(plot_density[plot_density!=0])))
        print("CLimitMax: " + str(cmax))



        





        if n_rows == 1:
            # ax_arr[col_i].imshow(plot_density, vmin=cmin, vmax=cmax)
            # ax_arr[col_i] = sns.heatmap(plot_density, vmin=cmin, vmax=cmax)

            c = ax_arr[col_i].pcolormesh(plot_density, vmin=cmin, vmax=cmax)

            if watershed is not None:
                ax_arr[col_i].plot(
                    watershed.borders[:, 0],
                    watershed.borders[:, 1],
                    ".k",
                    markersize=0.1,
                )
            ax_arr[col_i].set_aspect(0.9)
            ax_arr[col_i].set_title(label)
            ax_arr[col_i].set_xlabel("t-SNE 1")
            ax_arr[col_i].set_ylabel("t-SNE 2")
            ax_arr[col_i].set_xticks([])
            ax_arr[col_i].set_yticks([])
        else:
            # ax_arr[int(i / n_col), col_i].imshow(plot_density, vmin=cmin, vmax=cmax)
           
            # c = ax_arr[int(i / n_col), col_i].pcolormesh(plot_density, vmin=cmin, vmax=cmax)
            c = sns.heatmap(plot_density, vmin=cmin, vmax=cmax, ax=ax_arr[int(i / n_col), col_i], square=True)
            # c = sns.heatmap(plot_density, ax=ax_arr[int(i / n_col), col_i], robust=True, square=True)
            if watershed is not None:
                ax_arr[row_i, col_i].plot(
                    watershed.borders[:, 0],
                    watershed.borders[:, 1],
                    ".k",
                    markersize=0.1,
                )
            ax_arr[row_i, col_i].set_aspect(0.9)
            ax_arr[row_i, col_i].set_title(label)
            ax_arr[row_i, col_i].set_xlabel("t-SNE 1")
            ax_arr[row_i, col_i].set_ylabel("t-SNE 2")
            ax_arr[row_i, col_i].set_xticks([])
            ax_arr[row_i, col_i].set_yticks([])

    # f.colorbar(c, ax=ax_arr[row_i, col_i])        
    
    f.tight_layout()
    # f.colorbar(c, ax=ax_arr.ravel().tolist()) # need to set cbar position after calling tight_layout()
    plt.savefig(filepath, dpi=200)
    if show:
        plt.show()
    plt.close()
    return density, hist






def density_cat_norm(
    data: ds.DataStruct,
    column: str,
    watershed: Watershed,
    filepath: str = "./plot_folder/density_by_label.png",
    n_col: int = 4,
    transform = 'linear',
    cscale = 1, # used to define scale factor for visualization
    cmin = 0,
    cmax = None,
    show: bool = False,
):
    """
    Plot densities by a category label
    """
    labels = data.data[column].values
    n_col = min(n_col, len(np.unique(labels)))
    n_rows = int(np.ceil(len(np.unique(labels)) / n_col))
    f, ax_arr = plt.subplots(n_rows, n_col, figsize=((n_col + 1) * 4, n_rows * 4))

    # Loop over unique labels
    for i, label in enumerate(np.unique(labels)):
        embed_vals = data.embed_vals[data.data[column] == label]  # Indexing by label
        
        # density = watershed.fit_density(
        #     embed_vals, new=False
        # )  # Fit density on old axes
        
        ###################################
        # manual fit density
                # 2D histogram


        # coded hist functino in watershed seems problematic
        # new = False        
        # hist = watershed.hist(embed_vals, new)
        # density = hist/np.sum(hist)

        
        n_bins = 1000
        hist, xedges, yedges = np.histogram2d(
            embed_vals[:, 0],
            embed_vals[:, 1],
            bins=[n_bins, n_bins],
            # range=self.hist_range,
            density=False,
        )
        hist = np.rot90(hist)

        # density = hist

        density = hist/np.sum(hist)
        

        # density = gaussian_filter(density, sigma=watershed.sigma)

        # Calculates density using gaussian filter
        # density = gaussian_filter(hist, sigma=watershed.sigma)
        # # if self.log_out:
        # #     density = np.log1p(density)
        # density = np.clip(
        #     density, None, np.amax(density) * watershed.max_clip
        # )  # clips max for better visualization of clusters

        # if new:
        #     density = density

        ############################################
     
        col_i = i % n_col
        row_i = int(i/n_col)

        # scale density by number of data points (no. or rows) in category
        n_points = np.shape(embed_vals)[0]
        n_pixels = np.shape(density)[0]*np.shape(density)[1]

        
        
        # set scale to assume each point 

        # plot_density = density
        sigma=15
        plot_density = gaussian_filter(density, sigma=sigma)    
        # without gaussian filter, appearsnce of density seems to be correlated with number of samples  

        # plot_density = density

        # plot_density = density/n_points*n_pixels


        # cmax = n_points/n_pixels * cscale

        if(transform=='linear'):          
            plot_density = plot_density
        elif(transform=='softmax'):
            plot_density = scp.special.softmax(plot_density)
        elif(transform=='log2'):
            plot_density = np.log2(1+plot_density)
        elif(transform=='log2_ignore'):
            plot_density[plot_density!=0] = -np.log2(plot_density[plot_density!=0]) 
            plot_density[plot_density!=0] = plot_density[plot_density!=0]/-np.log2(cmax)     
        elif(transform=='log10'):
            plot_density = np.log10(1+plot_density)
        elif(transform=='log10_ignore'):
            plot_density[plot_density!=0] = -np.log10(plot_density[plot_density!=0]) 
            plot_density[plot_density!=0] = plot_density[plot_density!=0]/-np.log10(cmax)   
        elif(transform=='log10_before'):
            plot_density = np.log10(1+hist)/np.log10(1+cmax)      
        else:
            plot_density = plot_density        


        print("GroupID: " + str(label))
        print("No. of observations: " + str(n_points))
        print("No. of pixels in map: " + str(n_pixels))
        print("Sum of density: " + str(np.sum(plot_density)))
        print("No. of nonzero pixels: " + str(np.sum(plot_density!=0)))
        print("Max density (ignoring zeros): " + str(np.max(plot_density[plot_density!=0])))
        print("Mean density (ignoring zeros): " + str(np.mean(plot_density[plot_density!=0])))
        print("Median density (ignoring zeros): " + str(np.median(plot_density[plot_density!=0])))
        print("CLimitMax: " + str(cmax))



        





        if n_rows == 1:
            # ax_arr[col_i].imshow(plot_density, vmin=cmin, vmax=cmax)
            # ax_arr[col_i] = sns.heatmap(plot_density, vmin=cmin, vmax=cmax)

            c = ax_arr[col_i].pcolormesh(plot_density, vmin=cmin, vmax=cmax)

            if watershed is not None:
                ax_arr[col_i].plot(
                    watershed.borders[:, 0],
                    watershed.borders[:, 1],
                    ".k",
                    markersize=0.1,
                )
            ax_arr[col_i].set_aspect(0.9)
            ax_arr[col_i].set_title(label)
            ax_arr[col_i].set_xlabel("t-SNE 1")
            ax_arr[col_i].set_ylabel("t-SNE 2")
            ax_arr[col_i].set_xticks([])
            ax_arr[col_i].set_yticks([])
        else:
            # ax_arr[int(i / n_col), col_i].imshow(plot_density, vmin=cmin, vmax=cmax)
           
            # c = ax_arr[int(i / n_col), col_i].pcolormesh(plot_density, vmin=cmin, vmax=cmax)
            c = sns.heatmap(plot_density, vmin=cmin, vmax=cmax, ax=ax_arr[int(i / n_col), col_i], square=True)
            # c = sns.heatmap(plot_density, ax=ax_arr[int(i / n_col), col_i], robust=True, square=True)
            if watershed is not None:
                ax_arr[row_i, col_i].plot(
                    watershed.borders[:, 0],
                    watershed.borders[:, 1],
                    ".k",
                    markersize=0.1,
                )
            ax_arr[row_i, col_i].set_aspect(0.9)
            ax_arr[row_i, col_i].set_title(label)
            ax_arr[row_i, col_i].set_xlabel("t-SNE 1")
            ax_arr[row_i, col_i].set_ylabel("t-SNE 2")
            ax_arr[row_i, col_i].set_xticks([])
            ax_arr[row_i, col_i].set_yticks([])

    # f.colorbar(c, ax=ax_arr[row_i, col_i])        
    
    f.tight_layout()
    # f.colorbar(c, ax=ax_arr.ravel().tolist()) # need to set cbar position after calling tight_layout()
    plt.savefig(filepath, dpi=200)
    if show:
        plt.show()
    plt.close()
    return density, hist













def density_diff_norm(
    data: ds.DataStruct,
    column: str,
    watershed: Watershed,
    filepath: str = "./plot_folder/density_by_label.png",
    n_col: int = 4,
    transform = 'linear',
    cscale = 1, # used to define scale factor for visualization
    cmin = 0,
    cmax = None,
    show: bool = False,
):
    """
    Plot densities by a category label
    """
    labels = data.data[column].values
    n_col = min(n_col, len(np.unique(labels)))
    n_rows = int(np.ceil(len(np.unique(labels)) / n_col))
    f, ax_arr = plt.subplots(n_rows, n_col, figsize=((n_col + 1) * 4, n_rows * 4))

    # Loop over unique labels
    for i, label in enumerate(np.unique(labels)):
        embed_vals = data.embed_vals[data.data[column] == label]  # Indexing by label
        
        
        
        n_bins = 1000
        hist, xedges, yedges = np.histogram2d(
            embed_vals[:, 0],
            embed_vals[:, 1],
            bins=[n_bins, n_bins],
            density=False,
        )
        hist = np.rot90(hist)


        density = hist/np.sum(hist)
     
        col_i = i % n_col
        row_i = int(i/n_col)

        # scale density by number of data points (no. or rows) in category
        n_points = np.shape(embed_vals)[0]
        n_pixels = np.shape(density)[0]*np.shape(density)[1]

        # plot_density = density
        sigma=15
        plot_density = gaussian_filter(density, sigma=sigma)    
        # without gaussian filter, appearsnce of density seems to be correlated with number of samples  

        if(transform=='linear'):          
            plot_density = plot_density
        elif(transform=='softmax'):
            plot_density = scp.special.softmax(plot_density)
        elif(transform=='log2'):
            plot_density = np.log2(1+plot_density)
        elif(transform=='log2_ignore'):
            plot_density[plot_density!=0] = -np.log2(plot_density[plot_density!=0]) 
            plot_density[plot_density!=0] = plot_density[plot_density!=0]/-np.log2(cmax)     
        elif(transform=='log10'):
            plot_density = np.log10(1+plot_density)
        elif(transform=='log10_ignore'):
            plot_density[plot_density!=0] = -np.log10(plot_density[plot_density!=0]) 
            plot_density[plot_density!=0] = plot_density[plot_density!=0]/-np.log10(cmax)   
        elif(transform=='log10_before'):
            plot_density = np.log10(1+hist)/np.log10(1+cmax)      
        else:
            plot_density = plot_density        


        print("GroupID: " + str(label))
        print("No. of observations: " + str(n_points))
        print("No. of pixels in map: " + str(n_pixels))
        print("Sum of density: " + str(np.sum(plot_density)))
        print("No. of nonzero pixels: " + str(np.sum(plot_density!=0)))
        print("Max density (ignoring zeros): " + str(np.max(plot_density[plot_density!=0])))
        print("Mean density (ignoring zeros): " + str(np.mean(plot_density[plot_density!=0])))
        print("Median density (ignoring zeros): " + str(np.median(plot_density[plot_density!=0])))
        print("CLimitMax: " + str(cmax))

        if n_rows == 1:
            # ax_arr[col_i].imshow(plot_density, vmin=cmin, vmax=cmax)
            # ax_arr[col_i] = sns.heatmap(plot_density, vmin=cmin, vmax=cmax)

            c = ax_arr[col_i].pcolormesh(plot_density, vmin=cmin, vmax=cmax)

            if watershed is not None:
                ax_arr[col_i].plot(
                    watershed.borders[:, 0],
                    watershed.borders[:, 1],
                    ".k",
                    markersize=0.1,
                )
            ax_arr[col_i].set_aspect(0.9)
            ax_arr[col_i].set_title(label)
            ax_arr[col_i].set_xlabel("t-SNE 1")
            ax_arr[col_i].set_ylabel("t-SNE 2")
            ax_arr[col_i].set_xticks([])
            ax_arr[col_i].set_yticks([])
        else:
            # ax_arr[int(i / n_col), col_i].imshow(plot_density, vmin=cmin, vmax=cmax)
           
            # c = ax_arr[int(i / n_col), col_i].pcolormesh(plot_density, vmin=cmin, vmax=cmax)
            c = sns.heatmap(plot_density, vmin=cmin, vmax=cmax, ax=ax_arr[int(i / n_col), col_i], square=True)
            # c = sns.heatmap(plot_density, ax=ax_arr[int(i / n_col), col_i], robust=True, square=True)
            if watershed is not None:
                ax_arr[row_i, col_i].plot(
                    watershed.borders[:, 0],
                    watershed.borders[:, 1],
                    ".k",
                    markersize=0.1,
                )
            ax_arr[row_i, col_i].set_aspect(0.9)
            ax_arr[row_i, col_i].set_title(label)
            ax_arr[row_i, col_i].set_xlabel("t-SNE 1")
            ax_arr[row_i, col_i].set_ylabel("t-SNE 2")
            ax_arr[row_i, col_i].set_xticks([])
            ax_arr[row_i, col_i].set_yticks([])

    # f.colorbar(c, ax=ax_arr[row_i, col_i])        
    
    f.tight_layout()
    # f.colorbar(c, ax=ax_arr.ravel().tolist()) # need to set cbar position after calling tight_layout()
    plt.savefig(filepath, dpi=200)
    if show:
        plt.show()
    plt.close()
    return density, hist






# independent variables are 1) genotype, 2) age, 3) session (habituation vs baseline) 4) sex (male vs female)
# dependent variables are the clusters (this is a mutlivariate problem, multiple dependent variables)

# omnibus test: need an equivalent to multivariate 4-way anova
# post-hoc test: need mutli variate t-test (see Hotelling's t-test)

# mahalanobis distance (expoenent of multivariate normal dsitribution)
# https://online.stat.psu.edu/stat505/book/export/html/636

#what about two tailed for skewed/asymmetric distribution, is mahalanobis distance correct?




def bootstrap_twosample_multivariate(first_group, second_group, n_sampling=10000, statistic='mean'):
    # calculate p-value based on mahalabonis distance from center of null dsitribution(determined by mean)
    
    # This method only generates null distribution from the provided groups, but in our dataset we have more data for each cluster among groups

    # first_group, second_group: rows are independent samples, columns are multiple measurements (the dependent variables)


    first_group = np.squeeze(first_group)
    second_group = np.squeeze(second_group)

    first_metric = np.mean(first_group, axis=0)
    second_metric = np.mean(second_group, axis=0)
    difference_metric = second_metric - first_metric
    
    # def calculate_statistic(first_group, second_group, null_group, statistic):
    #     if statistic=='mean': # similar to Hotelling's Tsquared, (multivaraite t-test)
    #         first_metric = np.mean(first_group, axis=0)
    #         second_metric = np.mean(second_group, axis=0)
    #         difference_metric = second_metric - first_metric
    #     # elif: statistic=='F' # similar to multivariate ANOVA (MANOVA) # not fully implemented yet
    #     #     first_metric = np.sum(np.var(first_group, axis=0),axis=1)
    #     #     second_metric = np.sum(np.var(second_group, axis=0), axis=1)
    #     #     difference_metric = second_metric - first_metric  
    #     # elif statistic=='mahalabonis':
    #     #     # Invoke central limit theorem for null distribution of mean, can use mahalabonis distance to calculate p-value
    #     #     covariance_null = np.cov(bootstrap_null, rowvar=False)
    #     #     statistic_value  = np.matmul(np.matmul(difference_metric,np.linalg.inv(covariance_null)),difference_metric)
    #     # elif statistic=='hotelling':  
    #     #     covariance_null = np.cov(bootstrap_null, rowvar=False)
    #     #     statistic_value  = np.matmul(np.matmul(difference_metric,np.linalg.inv(covariance_null)),difference_metric)   

    first_size = np.shape(first_group)[0]
    second_size = np.shape(second_group)[0]

    if (np.shape(first_group)[1] == np.shape(second_group)[1]):
        n_clusters = np.shape(first_group)[1]
    else:
        print("Groups provided have different number of measurements")
        # return 
    
    merged = np.concatenate([first_group, second_group], axis=0)
    bootstrap_null = np.empty([n_sampling,n_clusters])

    rng = np.random.default_rng()

    print('Bootstrapping null distribution...' + "\n")
    for index_sampling in tqdm(range(n_sampling)):
        first_group_boot = rng.choice(merged, size=first_size, replace=True, p=None, axis=0, shuffle=False)
        second_group_boot = rng.choice(merged, size=second_size, replace=True, p=None, axis=0, shuffle=False)
        first_metric_boot = np.mean(first_group_boot, axis=0)
        second_metric_boot = np.mean(second_group_boot, axis=0)
        difference_metric_boot = second_metric_boot - first_metric_boot
        bootstrap_null[index_sampling,:] = difference_metric_boot

    # calculate p-value (two-tailed)
    p_value = np.empty([n_clusters])

    # Invoke central limit theorem for null distribution of mean, can use mahalabonis distance to calculate p-value
    # the differnece of 2 variables with gaussian distribution is still gaussian
    # # https://online.stat.psu.edu/stat500/book/export/html/572 
    mean_null = np.mean(bootstrap_null, axis=0)
    # print("Mean of null: " + str(mean_null))
    covariance_null = np.cov(bootstrap_null, rowvar=False)
    # covariance_null = np.cov(bootstrap_null.T, rowvar=False) # his might have cause singular matrix issue
    # for cnp.cov(), each row is a feature and each column is an observation


    # debug for singular matrix, covariance amtrix should always be positive definite so sohuld be invertable?
    if np.linalg.det(covariance_null)==0:
        print("Covariance matrix is singular")

        # return difference_metric, -1, bootstrap_null

        # inverting covariance amtrix seems to give singular value, trying using multivariate gau and that logs to get distance instead?
        # use Penrose pseudoinverse instead?
        # https://stats.stackexchange.com/questions/37743/singular-covariance-matrix-in-mahalanobis-distance-in-matlab
        # mahalabonis_distance_sample = np.matmul(np.matmul((difference_metric-mean_null),np.linalg.inv(covariance_null)),(difference_metric-mean_null))

        covariance_inverse_null = np.linalg.pinv(covariance_null)

        # mahalabonis_distance_sample = np.matmul(np.matmul((difference_metric-mean_null),np.linalg.pinv(covariance_null)),(difference_metric-mean_null))
    else:
        covariance_inverse_null = np.linalg.inv(covariance_null)
        # mahalabonis_distance_sample = np.matmul(np.matmul((difference_metric-mean_null),np.linalg.inv(covariance_null)),(difference_metric-mean_null))

    mahalabonis_distance_sample = np.matmul(np.matmul((difference_metric-mean_null),covariance_inverse_null),(difference_metric-mean_null))
    mahalabonis_distance_null = np.zeros(n_sampling)
    for index_sampling in tqdm(range(n_sampling)):
        mahalabonis_distance_null[index_sampling] = np.matmul(np.matmul((bootstrap_null[index_sampling,:]-mean_null),covariance_inverse_null),(bootstrap_null[index_sampling,:]-mean_null))
        # mahalabonis_distance_null[index_sampling] = np.matmul(np.matmul((bootstrap_null[index_sampling,:]-mean_null),np.linalg.inv(covariance_null)),(bootstrap_null[index_sampling,:]-mean_null))
        # mahalabonis_distance_null[index_sampling] = np.matmul(np.matmul((bootstrap_null[index_sampling,:]-mean_null),inv_lstsq(covariance_null)),(bootstrap_null[index_sampling,:]-mean_null))

    p_value = np.sum(mahalabonis_distance_null>=mahalabonis_distance_sample)/np.shape(bootstrap_null)[0]

    # calculate mahalabonis distance for null distribution data points
    # count frequency of datapoints  points greate than malahabonis distance of sample
    
        # fit boostrap data to multivariate normal

    # for index_cluster in tqdm(range(n_clusters)):
    #     # freq_greater = np.sum(bootstrap_null>=np.absolute(difference_metric))/np.shape(bootstrap_null)[0]
    #     # freq_lesser = np.sum(bootstrap_null<=-np.absolute(difference_metric))/np.shape(bootstrap_null)[0]
    #     # p_value[index_cluster] = freq_greater + freq_lesser
        
    #     p_value[index_cluster] = (np.sum(bootstrap_null[:,index_cluster]>=np.absolute(difference_metric.iloc[index_cluster])) + np.sum(bootstrap_null[:,index_cluster]<=-np.absolute(difference_metric.iloc[index_cluster])))/np.shape(bootstrap_null)[0]
    return difference_metric, p_value, bootstrap_null
    # return p_value

# # https://stackoverflow.com/questions/13795682/numpy-error-singular-matrix
# def inv_lstq(m):
#         a, b = m.shape
#         if a != b:
#             raise ValueError("Only square matrices are invertible.")

#         i = np.eye(a, a)
#     return np.linalg.lstsq(m, i)[0]


# from https://joernhees.de/blog/2015/08/26/scipy-hierarchical-clustering-and-dendrogram-tutorial/
def fancy_dendrogram(*args, **kwargs):
    max_d = kwargs.pop('max_d', None)
    if max_d and 'color_threshold' not in kwargs:
        kwargs['color_threshold'] = max_d
    annotate_above = kwargs.pop('annotate_above', 0)

    ddata = dendrogram(*args, **kwargs)

    if not kwargs.get('no_plot', False):
        plt.title('Hierarchical Clustering Dendrogram (truncated)')
        plt.xlabel('sample index or (cluster size)')
        plt.ylabel('distance')
        for i, d, c in zip(ddata['icoord'], ddata['dcoord'], ddata['color_list']):
            x = 0.5 * sum(i[1:3])
            y = d[1]
            if y > annotate_above:
                plt.plot(x, y, 'o', c=c)
                plt.annotate("%.3g" % y, (x, y), xytext=(0, -5),
                             textcoords='offset points',
                             va='top', ha='center')
        if max_d:
            plt.axhline(y=max_d, c='k')
    return ddata


# rewrite this function to allow breaking video into separate periods
def get_cluster_freq_by_cat_iloc(cluster_labels: np.ndarray, cat: np.ndarray):
    # cluster_labels = dff_Q1["Cluster"].values
    # cat=dff_Q1.id
    num_clusters = np.max(cluster_labels) + 1
    cat_labels = cat.iloc[np.sort(np.unique(cat, return_index=True)[1])] # changed cat[] to cat.iloc[]
    freq = np.zeros((len(cat_labels), num_clusters))
    for i, label in enumerate(tqdm(cat_labels)):
        # import pdb; pdb.set_trace()
        freq[i, :] =  np.histogram(
            cluster_labels[cat == label],
            bins=num_clusters,
            range=(-0.5, num_clusters - 0.5),
            density=True,
        )[0]
    return freq, cat_labels



# # added 2024/06/26 to flip axis (used to study if DANNCE can detect asymmetric/left vs right difference)
# def flip_axis(pose: np.ndarray, cat: np)