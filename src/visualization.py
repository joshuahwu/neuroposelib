import os
import numpy as np
import scipy.io as sio
import scipy as scp
import imageio
import tqdm
import hdf5storage

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
from typing import Optional, Union, List

import DataStruct as ds
from embed import Watershed, GaussDensity

palette = [(1,0.5,0),(0.5,0.5,0.85),(0,1,0),(1,0,0),(0,0,0.9),(0,1,1),
           (0.4,0.4,0.4),(0.5,0.85,0.5),(0.5,0.15,0.5),
           (0.15,0.5,0.5),(0.5,0.5,0.15),(0.9,0.9,0),(1,0,1),
           (0,0.5,1),(0.85,0.5,0.5),(0.5,1,0),(0.5,0,1),(1,0,0.5),(0,0.9,0.6),
           (0.3,0.6,0),(0,0.3,0.6),(0.6,0.3,0),(0.3,0,0.6),(0,0.6,0.3),(0.6,0,0.3)]

def scatter(data: Union[np.ndarray,ds.DataStruct],
            color: Optional[Union[List,np.ndarray]] = None,
            marker_size: int = 3,
            filepath: str = './plot_folder/scatter.png',
            show: bool = False,
            **kwargs):
    '''
    Draw a 2d tSNE plot from zValues.

    input: zValues dataframe, [num of points x 2]
    output: a scatter plot
    '''
    if isinstance(data, ds.DataStruct):
        x = data.embed_vals[:,0]
        y = data.embed_vals[:,1]
    elif isinstance(data, np.ndarray):
        if data.shape.index(2) == 1:
            x = data[:,0]
            y = data[:,1]
        else:
            x = data[0,:]
            y = data[1,:]

    f = plt.figure()
    plt.scatter(x, y, marker='.', s=marker_size, linewidths=0,
                c=color,cmap='YlOrRd', alpha=0.75, **kwargs)
    plt.xlabel('t-SNE 1')
    plt.ylabel('t-SNE 2')
    if color is not None:
        plt.colorbar()
    if filepath:
        plt.savefig(filepath,dpi=400)

    if show:
        plt.show()
    plt.close()
 

    # plt.figure(figsize=[12,10])
    # unique_animalID = np.unique(df_tSNE['animalID'])
    # for lbl in unique_animalID:
    #     plt.scatter(x=df_tSNE['x'][df_tSNE['animalID'] == lbl], 
    #                 y=df_tSNE['y'][df_tSNE['animalID'] == lbl], 
    #                 c=color, label=lbl, s=marker_size, **kwargs)
    # plt.legend()
    # plt.xlabel('t-SNE1')
    # plt.ylabel('t-SNE2')
    # plt.show()

def watershed(ws_map: np.ndarray,
              ws_borders: Optional[np.ndarray] = None,
              filepath: str='./plot_folder/watershed.png'):
    '''
    Plotting a watershed map with clusters colored

    '''
    f = plt.figure()
    ax = f.add_subplot(111)
    ax.imshow(ws_map)
    ax.set_aspect('auto')
    if ws_borders:
        ax.plot(ws_borders[:,0],ws_borders[:,1],'.r',markersize=0.05)
    plt.savefig(''.join([filepath,'_watershed.png']),dpi=400)
    plt.close()

def scatter_on_watershed(data: ds.DataStruct,
                         watershed: GaussDensity,
                         column: str,
                         ):
    labels = data.data[column].values

    if not os.path.exists(''.join([data.out_path,'points_by_cluster/'])):
        os.makedirs(''.join([data.out_path,'points_by_cluster/']))

    extent = [*watershed.hist_range[0], *watershed.hist_range[1]]

    f = plt.figure()
    ax = f.add_subplot(111)
    ax.imshow(watershed.watershed_map, zorder=1, extent=extent)
    ax.plot(data.embed_vals[:,0], data.embed_vals[:,1],
            '.r',markersize=1,alpha=0.1, zorder=2)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect('auto')
    filename = ''.join([data.out_path,'points_by_cluster/all.png'])
    plt.savefig(filename,dpi=400)
    plt.close()

    print("Plotting scatter on watershed for each ", column)
    for i, label in tqdm.tqdm(enumerate(np.unique(labels))):
        embed_vals = data.embed_vals[data.data[column]==label]

        f = plt.figure()
        ax = f.add_subplot(111)
        ax.imshow(watershed.watershed_map, zorder=0,
                  extent=extent)

        ax.plot(embed_vals[:,0], embed_vals[:,1],'.r',markersize=1,alpha=0.1, zorder=1)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect('auto')
        filename = ''.join([data.out_path,'points_by_cluster/',column,'_points_',str(label),'.png'])
        plt.savefig(filename,dpi=400)
        plt.close()

def density_feat(data: ds.DataStruct,
                 watershed: Watershed,
                 features: np.ndarray,
                 feature_labels: List,
                 key: str,
                 file_path: str = './plot_folder/'):
    
    feat_key = features[:,feature_labels.index(key)]
    density_feat = np.zeros((watershed.n_bins, watershed.n_bins))
    data_in_bin = watershed.map_bins(data.embed_vals)
    min_feat = np.min(feat_key)
    for i in tqdm.tqdm(range(watershed.n_bins)):
        for j in range(watershed.n_bins):
            bin_idx = np.logical_and(data_in_bin[:,0]==i, data_in_bin[:,1]==j)

            if np.all(bin_idx==False):
                density_feat[i,j] = min_feat
            else:
                density_feat[i,j] = np.mean(feat_key[bin_idx])

    density(density_feat,
            ws_borders = watershed.borders,
            filepath = ''.join([data.out_path,'density_feat_',key,'.png']),
            show=True)

def density(density: np.ndarray,
            ws_borders: Optional[np.ndarray] = None,
            filepath: str='./plot_folder/density.png',
            show: bool=False):
    f = plt.figure()
    ax = f.add_subplot(111)
    if ws_borders is not None:
        ax.plot(ws_borders[:,0],ws_borders[:,1],'.r',markersize=0.1)
    ax.imshow(density)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect('auto')
    if filepath:
        plt.savefig(filepath, dpi=400)
    if show:
        plt.show()
    plt.close()

def density_cat(data: ds.DataStruct,
                column: str,
                watershed: Watershed,
                filepath: str='./plot_folder/density_by_label.png',
                n_col: int=4,
                show: bool=False):
    '''
    Plot densities by a category label
    '''
    labels = data.data[column].values
    n_col = min(n_col, len(np.unique(labels)))
    n_rows = int(np.ceil(len(np.unique(labels))/n_col))
    f,ax_arr = plt.subplots(n_rows,n_col,
                            figsize=((n_col+1)*4,n_rows*4))

    # Loop over unique labels
    for i,label in enumerate(np.unique(labels)):
        embed_vals = data.embed_vals[data.data[column]==label] # Indexing by label
        density = watershed.fit_density(embed_vals,new=False) # Fit density on old axes
        if n_rows == 1:
            ax_arr[i%n_col].imshow(density)#scp.special.softmax(density))

            if watershed is not None:
                ax_arr[i%n_col].plot(watershed.borders[:,0], 
                                    watershed.borders[:,1],
                                    '.k', markersize=0.1)
            ax_arr[i%n_col].set_aspect('auto')
            ax_arr[i%n_col].set_title(label)
            ax_arr[i%n_col].set_xlabel('t-SNE 1')
            ax_arr[i%n_col].set_ylabel('t-SNE 2')
            ax_arr[i%n_col].set_xticks([])
            ax_arr[i%n_col].set_yticks([])
        else:
            ax_arr[int(i/n_col),i%n_col].imshow(scp.special.softmax(density))

            if watershed is not None:
                ax_arr[int(i/n_col),i%n_col].plot(watershed.borders[:,0], 
                                                watershed.borders[:,1],
                                                '.k', markersize=0.1)
            ax_arr[int(i/n_col),i%n_col].set_aspect('auto')
            ax_arr[int(i/n_col),i%n_col].set_title(label)
            ax_arr[int(i/n_col),i%n_col].set_xlabel('t-SNE 1')
            ax_arr[int(i/n_col),i%n_col].set_ylabel('t-SNE 2')
            ax_arr[int(i/n_col),i%n_col].set_xticks([])
            ax_arr[int(i/n_col),i%n_col].set_yticks([])
    f.tight_layout()
    plt.savefig(filepath, dpi=400)
    if show:
        plt.show()
    plt.close()
    return

def cluster_freq(data: ds.DataStruct,
                 conditions: List[str],
                 labels: List,
                 show = False):
    colors = ["tab:green","tab:blue","tab:orange","tab:red",
              "tab:purple","tab:brown","tab:pink","tab:gray",
              "tab:olive","tab:cyan","#dc0ab4","#00b7c7"]
    freq = data.cluster_freq()
    num_clusters = np.max(data.data['Cluster'])+1
    f,ax_arr = plt.subplots(len(conditions),1,sharex='all',figsize=(20,10))
    # import pdb; pdb.set_trace()
    for j in range(len(conditions)): # For each condition
        videos = data.meta.index[data.meta['Condition'] == conditions[j]].tolist()
        for i in range(len(videos)): # For each video
            ax_arr[j].plot(range(num_clusters), freq[videos[i],:], color=colors[i], label="video"+str(videos[i]))#''.join(['Animal',str(i)]))
        ax_arr[j].set_title(conditions[j],pad=-14)
        ax_arr[j].spines['top'].set_visible(False)
        ax_arr[j].get_xaxis().set_visible(False)
        ax_arr[j].spines['right'].set_visible(False)
        ax_arr[j].spines['bottom'].set_visible(False)
    ax_arr[1].set_ylabel('% Time Spent in Cluster')
    # ax_arr[0].legend(loc='upper right',ncol=6)
    # markers = ['o','v','s']
    # for j in range(len(conditions),len(conditions)+2): # For mean and variance plots
    #     for i in range(len(conditions)): #for each condition
    #         videos = data.meta.index[data.meta['Condition'] == conditions[j]].tolist()
    #         ax_arr[j].plot(range(num_clusters),np.mean(freq[,:],axis=1),color=colors[j],label=conditions[i], 
    #                         marker=markers[j], markersize=5,linewidth=0)
    #         # ax_arr[4].plot(range(num_clusters),np.std(freq[j,:,:],axis=1),color=colors[j],label=conditions[j], 
    #         #                 marker=markers[j], markersize=5, linewidth=0)
    #     ax_arr[j].spines['top'].set_visible(False)
    #     ax_arr[j].spines['right'].set_visible(False)

    # ax_arr[len(conditions)-2].legend(loc='upper right',ncol=3)
    # ax_arr[len(conditions)-2].spines['bottom'].set_visible(False)
    # ax_arr[len(conditions)-2].get_xaxis().set_visible(False)
    # ax_arr[len(conditions)-2].set_ylabel("Mean")
    # ax_arr[len(conditions)-1].set_ylabel("Standard Deviation")
    ax_arr[len(conditions)-1].set_xlabel('Cluster')
    ax_arr[len(conditions)-1].set_xlim([-0.25, 60.25])
    f.tight_layout()
    plt.subplots_adjust(hspace=0.05)
    plt.savefig(''.join([data.out_path,'mean_sd_cluster_freq.png']),dpi=400)
    if show:
        plt.show()

    plt.close()
    return

def skeleton_vid3D_cat(data: ds.DataStruct,
                       column: str,
                       labels: Optional[List] = None,
                       n_skeletons: int = 9):

    col_vals = data.data[column].values
    index = np.arange(len(col_vals))
    if labels is None:
        labels = np.unique(col_vals)

    for label in tqdm.tqdm(labels):
        label_idx = index[col_vals==label]
        if len(label_idx)==0:
            continue
        else:
            num_points = min(len(label_idx),n_skeletons)
            permuted_points = data.frame_id[np.random.permutation(label_idx)]# b/c moving frames filter
            sampled_points = []
            for i in range(len(permuted_points)):
                # import pdb; pdb.set_trace()
                if i == 0:
                    # print("first idx")
                    sampled_points = np.array([permuted_points[i]])
                    # print(sampled_points)
                    continue
                elif len(sampled_points)==num_points: # sampled enough points
                    break
                elif any(np.abs(permuted_points[i] - sampled_points)<200): # point is not far enough from previous points
                    continue
                else:
                    sampled_points = np.append(sampled_points,permuted_points[i])

            print(sampled_points)
            # import pdb; pdb.set_trace()
            skeleton_vid3D_expanded(data,
                                    label=label,
                                    frames = sampled_points,
                                    VID_NAME = ''.join([column,'_',str(label),'.mp4']),
                                    SAVE_ROOT = ''.join([data.out_path, 'skeleton_vids/']))


def skeleton_vid3D_expanded(data: Union[ds.DataStruct, np.ndarray],
                            label: str,
                            connectivity: Optional[ds.Connectivity]=None,
                            frames: List = [3000,100000,5000000], 
                            N_FRAMES: int = 150,
                            fps: int = 90,
                            dpi: int = 100,
                            VID_NAME: str = '0.mp4',
                            SAVE_ROOT: str = './test/skeleton_vids/'):
    

    if isinstance(data, ds.DataStruct):
        preds = data.pose_3d
        connectivity = data.connectivity
    else:
        preds = data

    if connectivity is None:
        skeleton_name = 'mouse' + str(preds.shape[1])
        connectivity = Connectivity().load('../../CAPTURE_data/skeletons.py',
                                           skeleton_name=skeleton_name)

    START_FRAME = np.array(frames) - int(N_FRAMES/2) + 1
    COLOR = connectivity.colors*len(frames)
    links = connectivity.links
    links_expand = links
    # total_frames = N_FRAMES*len(frames)#max(np.shape(f[list(f.keys())[0]]))

    ## Expanding connectivity for each frame to be visualized
    num_joints = max(max(links))+1
    for i in range(len(frames)-1):
        next_con = [(x+(i+1)*num_joints, y+(i+1)*num_joints) for x,y in links]
        links_expand=links_expand+next_con

    save_path = os.path.join(SAVE_ROOT)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # get dannce predictions
    pose_3d = np.empty((0, num_joints, 3))
    for start in START_FRAME:
        pose_3d = np.append(pose_3d, preds[start:start+N_FRAMES,:,:],axis=0)

    # compute 3d grid limits 
    offset = 50
    x_lim1, x_lim2 = np.min(pose_3d[:, :, 0])-offset, np.max(pose_3d[:, :, 0])+offset
    y_lim1, y_lim2 = np.min(pose_3d[:, :, 1])-offset, np.max(pose_3d[:, :, 1])+offset
    z_lim1, z_lim2 = np.minimum(0, np.min(pose_3d[:, :, 2])), np.max(pose_3d[:, :, 2])+10

    # set up video writer
    metadata = dict(title='dannce_visualization', artist='Matplotlib')
    writer = FFMpegWriter(fps=fps)#, metadata=metadata)

    extent = [*data.ws.hist_range[0], *data.ws.hist_range[1]]
    embed_vals = data.embed_vals[data.data['Cluster']==label]
    colors = ["k","tab:blue","tab:green","tab:orange",
              "tab:purple","tab:brown","tab:pink","tab:gray",
              "tab:olive","tab:cyan","#dc0ab4","#00b7c7"]

    # Setup figure
    fig = plt.figure(figsize=(20, 10))
    gs = fig.add_gridspec(1, 2)
    ax_3d = fig.add_subplot(gs[0,1], projection='3d')
    ax_dens = fig.add_subplot(gs[0,0])

    cond_uniq = data.data['Condition'].unique()
    frames_meta = data.data[data.data['frame_id'].isin(frames)]['Condition'].values

    ax_dens.imshow(data.ws.watershed_map, zorder=0,
                    extent=extent)
    ax_dens.plot(embed_vals[:,0], embed_vals[:,1],'.r',markersize=1,alpha=0.1, zorder=1)
    ax_dens.set_aspect('auto')
    ax_dens.set_xticks([])
    ax_dens.set_yticks([])

    with writer.saving(fig, os.path.join(save_path, "vis_"+VID_NAME), dpi=dpi):

        for curr_frame in tqdm.tqdm(range(N_FRAMES)):
            # grab frames
            curr_frames = curr_frame + np.arange(len(frames))*N_FRAMES
            kpts_3d = np.reshape(pose_3d[curr_frames,:,:], (len(frames)*num_joints, 3))
            
            # plot 3d moving skeletons
            for i in range(len(np.unique(frames_meta))):
                temp_idx = curr_frames[frames_meta==np.unique(frames_meta)[i]]
                temp_kpts = np.reshape([pose_3d[temp_idx,:,:]], (len(temp_idx)*num_joints,3))
                ax_3d.scatter(temp_kpts[:, 0], temp_kpts[:, 1], temp_kpts[:, 2],  marker='.', color=colors[i], linewidths=0.5, label=cond_uniq[i])
            ax_3d.legend()
            for color, (index_from, index_to) in zip(COLOR, links_expand):
                xs, ys, zs = [np.array([kpts_3d[index_from, j], kpts_3d[index_to, j]]) for j in range(3)] 
                ax_3d.plot3D(xs, ys, zs, c=color, lw=2)

            ax_3d.set_xlim(x_lim1, x_lim2)
            ax_3d.set_ylim(y_lim1, y_lim2)
            ax_3d.set_zlim(0, 150)
            ax_3d.set_xlabel("x")
            ax_3d.set_ylabel("y")
            # ax_3d.set_xticks([])
            # ax_3d.set_yticks([])
            # ax_3d.set_zticks([])
            # ax_3d.set_title("3D Tracking")
            # ax_3d.set_aspect('equal')
            ax_3d.set_box_aspect([1,1,0.4])

            # grab frame and write to vid
            writer.grab_frame()
            fig.tight_layout()
            ax_3d.clear()
    
    plt.close()
    return 0

def skeleton_vid3D(data: Union[ds.DataStruct, np.ndarray],
                   connectivity: Optional[ds.Connectivity]=None,
                   frames: List = [3000,100000,500000], 
                   N_FRAMES: int = 300,
                   fps: int = 90,
                   dpi: int = 200,
                   VID_NAME: str = '0.mp4',
                   SAVE_ROOT: str = './test/skeleton_vids/'):

    if isinstance(data, ds.DataStruct):
        preds = data.pose_3d
        connectivity = data.connectivity
    else:
        preds = data

    if connectivity is None:
        skeleton_name = 'mouse' + str(preds.shape[1])
        connectivity = Connectivity().load('../../CAPTURE_data/skeletons.py',
                                           skeleton_name=skeleton_name)

    START_FRAME = np.array(frames) - int(N_FRAMES/2) + 1
    COLOR = connectivity.colors*len(frames)
    links = connectivity.links
    links_expand = links
    # total_frames = N_FRAMES*len(frames)#max(np.shape(f[list(f.keys())[0]]))

    ## Expanding connectivity for each frame to be visualized
    num_joints = max(max(links))+1
    for i in range(len(frames)-1):
        next_con = [(x+(i+1)*num_joints, y+(i+1)*num_joints) for x,y in links]
        links_expand=links_expand+next_con

    save_path = os.path.join(SAVE_ROOT)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # get dannce predictions
    pose_3d = np.empty((0, num_joints, 3))
    for start in START_FRAME:
        pose_3d = np.append(pose_3d, preds[start:start+N_FRAMES,:,:],axis=0)

    # compute 3d grid limits 
    offset = 50
    x_lim1, x_lim2 = np.min(pose_3d[:, :, 0])-offset, np.max(pose_3d[:, :, 0])+offset
    y_lim1, y_lim2 = np.min(pose_3d[:, :, 1])-offset, np.max(pose_3d[:, :, 1])+offset
    z_lim1, z_lim2 = np.minimum(0, np.min(pose_3d[:, :, 2])), np.max(pose_3d[:, :, 2])+10

    # set up video writer
    # metadata = dict(title='dannce_visualization', artist='Matplotlib')
    writer = FFMpegWriter(fps=fps)#, metadata=metadata)

    # Setup figure
    fig = plt.figure(figsize=(12, 12))
    ax_3d = fig.add_subplot(1, 1, 1, projection='3d')
    with writer.saving(fig, os.path.join(save_path, "vis_"+VID_NAME), dpi=dpi):
        for curr_frame in tqdm.tqdm(range(N_FRAMES)):
            # grab frames
            curr_frames = curr_frame + np.arange(len(frames))*N_FRAMES
            kpts_3d = np.reshape(pose_3d[curr_frames,:,:], (len(frames)*num_joints, 3))
            
            # plot 3d moving skeletons
            ax_3d.scatter(kpts_3d[:, 0], kpts_3d[:, 1], kpts_3d[:, 2],  marker='.', color='black', linewidths=0.5)
            for color, (index_from, index_to) in zip(COLOR, links_expand):
                xs, ys, zs = [np.array([kpts_3d[index_from, j], kpts_3d[index_to, j]]) for j in range(3)] 
                ax_3d.plot3D(xs, ys, zs, c=color, lw=2)

            ax_3d.set_xlim(x_lim1, x_lim2)
            ax_3d.set_ylim(y_lim1, y_lim2)
            ax_3d.set_zlim(0, 150)
            ax_3d.set_title("3D Tracking")
            ax_3d.set_xlabel("x")
            ax_3d.set_ylabel("y")
            # ax_3d.set_aspect('equal')
            ax_3d.set_box_aspect([1,1,0.4])

            # grab frame and write to vid
            writer.grab_frame()
            ax_3d.clear()
    
    plt.close()
    return 0


def skeleton_vid3D_features(pose,
                            feature,
                            connectivity: Optional[ds.Connectivity]=None,
                            frames: List = [3000], 
                            N_FRAMES: int = 150,
                            fps: int = 90,
                            dpi: int = 200,
                            VID_NAME: str = '0.mp4',
                            SAVE_ROOT: str = './test/skeleton_vids/'):

    START_FRAME = np.array(frames) - int(N_FRAMES/2) + 1
    COLOR = connectivity.colors*len(frames)
    links = connectivity.links
    links_expand = links

    ## Expanding connectivity for each frame to be visualized
    num_joints = max(max(links))+1
    for i in range(len(frames)-1):
        next_con = [(x+(i+1)*num_joints, y+(i+1)*num_joints) for x,y in links]
        links_expand=links_expand+next_con

    save_path = os.path.join(SAVE_ROOT)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # get dannce predictions
    pose_3d = np.empty((0, num_joints, 3))
    for start in START_FRAME:
        pose_3d = np.append(pose_3d, pose[start:start+N_FRAMES,:,:],axis=0)
        feature = feature[start:start+N_FRAMES]

    # compute 3d grid limits 
    offset = 10
    x_lim1, x_lim2 = np.min(pose_3d[:, :, 0])-offset, np.max(pose_3d[:, :, 0])+offset
    y_lim1, y_lim2 = np.min(pose_3d[:, :, 1])-offset, np.max(pose_3d[:, :, 1])+offset
    z_lim1, z_lim2 = np.minimum(0, np.min(pose_3d[:, :, 2])), np.max(pose_3d[:, :, 2])+10

    # set up video writer
    writer = FFMpegWriter(fps=int(fps/4))

    # Setup figure
    fig = plt.figure(figsize=(20, 10))
    gs = fig.add_gridspec(1, 2)
    ax_3d = fig.add_subplot(gs[0,1], projection='3d')
    ax_trace = fig.add_subplot(gs[0,0])

    with writer.saving(fig, os.path.join(save_path, "vis_feat_"+VID_NAME), dpi=dpi):

        for curr_frame in tqdm.tqdm(range(N_FRAMES)):
            # grab frames
            curr_frames = curr_frame + np.arange(len(frames))*N_FRAMES

            ax_trace.plot(curr_frames,feature[curr_frames],marker='.', markersize=20, color='k')

            kpts_3d = np.reshape(pose_3d[curr_frames,:,:], (len(frames)*num_joints, 3))

            # plot 3d moving skeletons
            ax_3d.scatter(kpts_3d[:, 0], kpts_3d[:, 1], kpts_3d[:, 2],  marker='.', color='black', linewidths=0.5)
            for color, (index_from, index_to) in zip(COLOR, links_expand):
                xs, ys, zs = [np.array([kpts_3d[index_from, j], kpts_3d[index_to, j]]) for j in range(3)] 
                ax_3d.plot3D(xs, ys, zs, c=color, lw=2)

            ax_3d.set_xlim(x_lim1, x_lim2)
            ax_3d.set_ylim(y_lim1, y_lim2)
            ax_3d.set_zlim(0, 150)
            ax_3d.set_xlabel("x")
            ax_3d.set_ylabel("y")
            # ax_3d.set_xticks([])
            # ax_3d.set_yticks([])
            # ax_3d.set_zticks([])
            # ax_3d.set_title("3D Tracking")
            # ax_3d.set_aspect('equal')
            ax_3d.set_box_aspect([1,1,0.4])

            # grab frame and write to vid
            writer.grab_frame()
            fig.tight_layout()
            ax_3d.clear()
    
    plt.close()
    return 0


# def draw_tSNE_interactive(self, df_tSNE, color='animalID', marker_size=3):
#     '''
#     Draw an interactive 2d tSNE plot from zValues.

#     input: zValues dataframe, [num of points x 2]
#     output: a scatter plot
#     '''
#     unique_animalID = np.unique(df_tSNE['animalID'])
#     fig = px.scatter(df_tSNE, x='x', y='y', color=color, hover_data=['idx', 'x', 'y'], width=800, height=800)
#     fig.update_traces(marker_size=marker_size)
#     fig.show()

# def draw_3d_skeleton(self, selected_anchor_idx):
#     '''
#     input: an index of predictions 3d coordinates
#     output: a skeleton scatter3d plot of that index
#     '''
#     selected_anchor_idx = int(selected_anchor_idx)
#     temp_list = []
#     anchors = list(predictions_dict['predictions'][0][0])[:-1] # 'Tail_base_' etc. excluding last matrix 'sampleID'
#     for i in range(len(anchors)): 
#         temp_list.append(anchors[i][selected_anchor_idx])
#     plt.figure()
#     ax = plt.axes(projection="3d")
#     for x, y, z in temp_list:
#         ax.scatter3D(x, y, z)
#     skeleton_color, colors = None, self.left_or_right_colormap_dict['color'].tolist()
#     for i, (first, second) in enumerate(self.left_or_right_colormap_dict['joints_idx']):
#         xx = [anchors[first-1][selected_anchor_idx][0], anchors[second-1][selected_anchor_idx][0]]
#         yy = [anchors[first-1][selected_anchor_idx][1], anchors[second-1][selected_anchor_idx][1]]
#         zz = [anchors[first-1][selected_anchor_idx][2], anchors[second-1][selected_anchor_idx][2]]
#         if colors[i] == [1, 0, 0]:
#             skeleton_color = 'r' 
#         elif colors[i] == [0, 1, 0]:
#             skeleton_color = 'g'
#         else:
#             skeleton_color = 'b'
#         ax.plot(xx, yy, zz, c=skeleton_color)
#     plt.show()

# def draw_3d_skeleton_interactive(self, selected_anchor_idx, marker_size=3):
#     '''
#     input: an index of predictions 3d coordinates
#     output: a skeleton scatter3d plot of that index
#     '''
#     selected_anchor_idx = int(selected_anchor_idx)
#     temp_list = []
#     anchors = list(predictions_dict['predictions'][0][0])[:-1] # 'Tail_base_' etc. excluding last matrix 'sampleID'
#     for i in range(len(anchors)): 
#       temp_list.append(anchors[i][selected_anchor_idx])
#     df_temp = pd.DataFrame(temp_list)
#     print(df_temp.head())
#     fig = px.scatter_3d(temp_list, x=0, y=1, z=2)
#     skeleton_color, colors = None, self.left_or_right_colormap_dict['color'].tolist()
#     fig = go.Figure()
#     for i, (first, second) in enumerate(self.left_or_right_colormap_dict['joints_idx']):
#         xx = [anchors[first-1][selected_anchor_idx][0], anchors[second-1][selected_anchor_idx][0]]
#         yy = [anchors[first-1][selected_anchor_idx][1], anchors[second-1][selected_anchor_idx][1]]
#         zz = [anchors[first-1][selected_anchor_idx][2], anchors[second-1][selected_anchor_idx][2]]
#         name = self.left_or_right_colormap_dict['joint_names'][0][first-1].tolist()[0]+'-'\
#                 + self.left_or_right_colormap_dict['joint_names'][0][second-1].tolist()[0]
#         if colors[i] == [1, 0, 0]:
#             skeleton_color = 'r' 
#         elif colors[i] == [0, 1, 0]:
#             skeleton_color = 'g'
#         else:
#             skeleton_color = 'b'
#         fig.add_scatter3d(x=xx, y=yy, z=zz, name=name)
#     fig.update_traces(marker_size=marker_size)
#     fig.show()

# def path_format_switch(self, original_path):
#     '''
#     Switches the path from format
#     '/media/twd/dannce-pd/PDBmirror/2021-07-04-PDb1_0-dopa'
#     to
#     '/hpc/group/tdunn/pdb_data/videos/2021_04_07/PDb1_R1_0/videos/'
#     '''
#     year, day, month, pdb, _ = original_path.split('/')[5].split('-')
#     pdb1, pdb2 = pdb[3], pdb[5]
#     converted_path = self.common_path + '/videos/{}_{}_{}/PDb{}_R1_{}/videos'.format(year, month, day, pdb1, pdb2)
#     return converted_path

# def get_relative_frame_idx(self, overall_idx):
#     '''
#     Get the index in a certain video from the overall index. 

#     input: Overall index.
#     output: frame_number: relative index in a video. v
#             id_idx: which video is the frame in. 
#             amount_of_frames: amount of frames per video (assuming each video has same length)
#     '''
#     overall_idx = int(overall_idx)
#     example_video = cv2.VideoCapture(self.videopaths[1])
#     amount_of_frames = int(example_video.get(cv2.CAP_PROP_FRAME_COUNT))
#     print('Frames per video:', amount_of_frames)
#     frame_number = overall_idx % amount_of_frames
#     vid_idx = overall_idx // amount_of_frames + 1 # find which vid the frame belongs to. E.g., 3 // 324000 + 1 = 0 + 1 = 1. So frame 3 is in 1st video.
#     print('The requested frame idx {} is in video number {}, at the {}th frame of that video.'.format(overall_idx, vid_idx, frame_number))
#     return frame_number, vid_idx, amount_of_frames

#   def frame2time(self, overall_idx):
#     '''
#     Convert the overall index of a frame (a point in tSNE) to the time stamp (in second).

#     input: overall_idx, the index of the frame in all videos (0-2M)
#     output: timestamp of the frame in seconds.
#     '''
#     frame_number, vid_idx, amount_of_frames = self.get_relative_frame_idx(overall_idx)
#     return frame_number / amount_of_frames * self.vid_length # the length of vid is 3600s.

#   def get_frame(self, overall_idx):
#     '''
#     Get the frame in the set of videos with a given index.
#     For example, there are 7 videos, each with 324000 frames,
#     so there are 324000x7=2268000 frames. You can give
#     overall_idx = 2000001

#     input: overall_idx, the index of the frame in all videos (0-2M)
#     output: an .jpg image of that frame
#     '''
#     print('Starts extracting frame...')
#     frame_number, vid_idx, amount_of_frames = self.get_relative_frame_idx(overall_idx)
#     video = cv2.VideoCapture(self.videopaths[vid_idx])
#     video.set(cv2.CAP_PROP_POS_FRAMES, frame_number-1)
#     is_success, frame = video.read()
#     frame = cv2.resize(frame, (500, 500))
#     cv2_imshow(frame)
#     if is_success:
#         image_name = "video{}_frame{}.jpg".format(vid_idx, frame_number)
#         cv2.imwrite(image_name, frame)
#         print('Frame successfully extracted as', image_name)

# def generate_video(self, overall_idx, video_length=3000):
#     '''
#     Generate a video around the given frame of length video_length (in ms).

#     input: overall frame index，desired video length (e.g., 3000ms)
#     output: a video fraction
#     '''
#     print('Starts generating video...')
#     frame_number, vid_idx, amount_of_frames = self.get_relative_frame_idx(overall_idx)
#     timestamp = self.frame2time(overall_idx)
#     half_length = video_length / 2000 # in seconds
#     start, end = max(timestamp - half_length, 0), min(timestamp + half_length, self.vid_length)
#     video_name = "video{}_time{}m{}s.mp4".format(vid_idx, round(timestamp // 60, 2), round(timestamp % 60, 2))
#     ffmpeg_extract_subclip(anst.videopaths[vid_idx], start, end, targetname=video_name)
#     print('Video clip {} generated successfully with length of {} secs.'.format(video_name, round(end - start, 2)))