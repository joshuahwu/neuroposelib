from neuroposelib import read
from neuroposelib import vis
from neuroposelib import preprocess
from neuroposelib import write
from neuroposelib import features
from neuroposelib import analysis
from neuroposelib import DataStruct as ds
from neuroposelib.embed import Embed
from neuroposelib.embed import Watershed
import pandas as pd
import numpy as np


def get_data_obj(
                ego_pose, labels, config, pose, ids, meta, meta_by_frame, connectivity,
                categories = ["ego_euc"], 
                n_pcs = 5, 
                method="fbpca", 
                fps = 90,
                ):
    from neuroposelib import DataStruct as dsnew
    pc_feats, pc_labels = features.pca(
                                        ego_pose, labels, categories=["ego_euc"], n_pcs=n_pcs, method=method
                                    )

    del ego_pose, labels

    wlet_feats, wlet_labels = features.wavelet(
        pc_feats, pc_labels, ids, fs=fps, freq=np.linspace(1, 25, 25), bw=5
    )

    # PCA on wavelet features
    pc_wlet, pc_wlet_labels = features.pca(
        wlet_feats,
        wlet_labels,
        categories=["wlet_ego_euc"],
        # categories=["wlet_ang"],
        n_pcs=n_pcs,
        method=method,
    )

    del wlet_feats, wlet_labels
    pc_feats = np.hstack((pc_feats, pc_wlet))
    pc_labels += pc_wlet_labels
    del pc_wlet, pc_wlet_labels

    data_obj = dsnew.DataStruct(
        pose=           pose,
        id=             ids,
        meta=           meta,
        meta_by_frame=  meta_by_frame,
        connectivity=   connectivity,
    )
    embedder = Embed(
        embed_method=   config["single_embed"]["method"],
        perplexity=     config["single_embed"]["perplexity"],
        lr=             config["single_embed"]["lr"],
    )
    
    data_obj.features = pc_feats
    data_obj = data_obj[:: config["downsample"], :]
    data_obj.embed_vals = embedder.embed(data_obj.features, save_self=True)
    data_obj.ws = Watershed(
        sigma=config["single_embed"]["sigma"], max_clip=1, log_out=True, pad_factor=0.05
    )
    data_obj.data["Cluster"] = data_obj.ws.fit_predict(data=data_obj.embed_vals)

    return data_obj


def populate_new_data_obj(data_obj, config, 
                            pose = None, 
                            ids = None, 
                            meta = None, 
                            meta_by_frame = None, 
                            connectivity = None
                        ):
    '''
    Populate a new data object by performing clustering and embedding again on the features extracted from
    the provided data object
    '''
    from dappy import DataStruct as dsnew

    if pose is None and ids is None and meta is None and meta_by_frame is None and connectivity is None:
        new_data_obj = copy.deepcopy(data_obj)
    else:
        new_data_obj = dsnew.DataStruct(
            pose=           pose,
            id=             ids,
            meta=           meta,
            meta_by_frame=  meta_by_frame,
            connectivity=   connectivity,
        )
        new_data_obj = new_data_obj[:: config["downsample"], :]
        new_data_obj.features = data_obj.features

    embedder = Embed(
        embed_method=   config["single_embed"]["method"],
        perplexity=     config["single_embed"]["perplexity"],
        lr=             config["single_embed"]["lr"],
    )

    '''
        # This embedding seems to be the key step. After this the cluster numbers and assignments do not change
        # evs_2 = body_embedder.embed(data_obj_1.features, save_self=True)
        # ws_2 = copy.deepcopy(data_obj_1.ws)
        # clusters_2 = ws_2.fit_predict(data=evs_2)
        # clusters_3 = ws_2.fit_predict(data=evs_2)
        # np.isclose(clusters_2, clusters_3).all() -> Gives True

    '''
    
    new_data_obj.embed_vals = embedder.embed(data_obj.features, save_self=True)
    new_data_obj.ws = Watershed(
        sigma=config["single_embed"]["sigma"], max_clip=1, log_out=True, pad_factor=0.05
    )
    new_data_obj.data["Cluster"] = new_data_obj.ws.fit_predict(data=new_data_obj.embed_vals)

    return new_data_obj


def get_clust_frame_groupings(data_obj_1):
    df_1 = data_obj_1.data
    df_1_clust_frame = df_1.groupby('Cluster')['frame'].apply(np.array).reset_index().set_index('Cluster')

    return df_1_clust_frame

def best_matches(df1, df2):
    #TODO: Consider using pandas crosstab function
    # match_results = pd.DataFrame(index=df1.index, columns=df1.columns)
    match_results = df1.copy(deep=True)
    
    for df1_index, row in df1.iterrows():
        for col, arr1 in row.items():
            best_match = None
            best_intersection = 0
            best_intersection_len = 0
            if isinstance(arr1, str):
                arr1 = np.fromstring(arr1.strip('[]'), sep=' ', dtype='int')
            
            for df2_index, row2 in df2.iterrows():
                arr2 = row2[col]

                if isinstance(arr2, str):
                    arr2 = np.fromstring(arr2.strip('[]'), sep=' ', dtype='int')
                
                intersection_size = np.intersect1d(arr1, arr2).size
                print(intersection_size, arr1.dtype, arr2.dtype)
                
                if intersection_size > best_intersection:
                    best_intersection = intersection_size
                    best_match = df2_index
                    best_intersection_len = arr2.size
                    # print ('Best Intersection Size = ', best_intersection_len)
            
            match_results.loc[df1_index, "ClusterMatch"] = best_match
            if best_match:
                match_results.loc[df1_index, "MatchFrom%"] = (best_intersection/arr1.size)*100
                match_results.loc[df1_index, "MatchTo%"] = (best_intersection/best_intersection_len)*100
            else:
                match_results.loc[df1_index, "MatchFrom%"] = 100.0
                match_results.loc[df1_index, "MatchTo%"] = 0.0
            
            
    
    return match_results


def get_clust_frame(ego_pose=None, labels=None, config=None, pose=None, ids=None, meta=None, meta_by_frame=None, connectivity=None,
                    data_obj=None,
                    categories = ["ego_euc"], 
                    n_pcs = 5, 
                    method="fbpca", 
                    fps = 90,
                    ):
    
    if ego_pose is None         and \
        data_obj is not None    :

        dobj = populate_new_data_obj(data_obj, config, pose, ids, meta, meta_by_frame, connectivity)
    else:
        dobj = get_data_obj(  ego_pose, 
                            labels, 
                            config, 
                            pose, 
                            ids, 
                            meta, 
                            meta_by_frame, 
                            connectivity,
                        )
    
    return get_clust_frame_groupings(dobj), dobj.data



