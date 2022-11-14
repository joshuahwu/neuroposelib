import pandas as pd
import numpy as np
import h5py
import hdf5storage
import matplotlib.pyplot as plt
from typing import Optional, Union, List, Tuple
import yaml

class DataStruct:

    '''
    Class for organizing and linking metadata to features and pose data
    '''

    _props = [
        'embed_vals',
        'data',
        'meta_by_frame',
        'frame',
        'features',
        'meta',
    ]

    def __init__(self,
                 data: pd.DataFrame = pd.DataFrame(),
                 id: Optional[Union[List, np.ndarray]] = None,
                 meta: pd.DataFrame = pd.DataFrame(),
                 meta_by_frame: pd.DataFrame = pd.DataFrame(),
                 id_full: Optional[Union[List, np.ndarray]] = None,
                 pose: np.ndarray = None,
                 connectivity = None,
                 frame: Optional[Union[List, np.ndarray]] = None,
                 feature_labels: Optional[List[str]] = None):
        
        self.data = data
        self.meta = meta
        self.connectivity = connectivity
        self.meta_by_frame = meta_by_frame
        
        if (id is not None) or ('id' not in self.data):
            self.id = id

        self.pose = pose
        self.feature_labels = feature_labels

        if frame is not None:
            self.frame = frame
        elif 'frame' not in self.data:
            self.frame = np.arange(0, self.data.shape[0])

        if id_full is not None:
            self.id_full = id_full
        else:
            self.id_full = self.id
        
        # import pdb; pdb.set_trace()

    def __getitem__(self, idx):
        if not isinstance(idx, tuple):
            idx=tuple(idx)
        new_struct = DataStruct(data = self.data.loc[idx].reset_index(drop=True),
                                pose = self.pose,
                                meta = self.meta,
                                id_full = self.id_full,
                                feature_labels = self.feature_labels,
                                connectivity = self.connectivity)
        return new_struct
        
    def check_reset_data(self, len:int):
        if self.data.shape[0] != len:
            self.data = pd.DataFrame()

    @property
    def frame(self):
        return self.data['frame'].to_numpy()

    @frame.setter
    def frame(self,
                 frame: Union[List[int],np.array]):
        self.data['frame'] = frame
    
    @property
    def id(self):
        return self.data['id'].to_numpy()

    @id.setter
    def id(self,
               id: Union[List[Union[str, int]],np.array]):
        self.data['id'] = id

    @property
    def meta_by_frame(self):
        return self.data[list(self.meta.columns.values)]

    @meta_by_frame.setter
    def meta_by_frame(self,
                      meta_by_frame: pd.DataFrame):
        self.data.loc[:, list(meta_by_frame.columns.values)] = meta_by_frame.values

    def meta_unique(self,
                    column_id: str):
        return list(set(list(self.data[column_id].values)))

    @property
    def n_frames(self):
        return self.data.shape[0]

    @property
    def embed_vals(self):
        return np.array(list(self.data['embed_vals'].to_numpy()))

    @embed_vals.setter
    def embed_vals(self,
                   embed_vals: Optional[np.array]=None):
        self.data['embed_vals'] = list(embed_vals)

    @property
    def features(self):
        return np.array(list(self.data['features'].to_numpy())) # seems slow

    @features.setter
    def features(self,
                 features: np.array):
        self.data['features'] = list(features) # seems slow

    @property
    def feat_shape(self):
        return np.shape(self.features)

    def write_pickle(self,
                     out_path: Optional[str] = None):
        # if out_path is None:
        #     out_path = self.out_path
        import pickle
        pickle.dump(self, open(''.join([out_path,"datastruct.p"]), "wb"))

    def downsample(self,
                   downsample: int):
        return self[::downsample]

    def cluster_freq(self):
        '''
        Calculates the percentage of time each id spends in each cluster
        OUT:
            freq: numpy array of (# videos x # clusters)
        '''
        num_clusters = np.max(self.data['Cluster'])+1
        freq = np.zeros((len(self.meta.index), num_clusters))
        for i in self.meta.index:
            cluster_by_exp = self.data['Cluster'][self.data['id']==i]
            freq[i,:] = np.histogram(cluster_by_exp, bins=num_clusters, range=(-0.5,num_clusters-0.5))[0]
        frame_totals = np.sum(freq,axis=1)
        frame_totals = np.where(frame_totals==0, 1, frame_totals)
        freq = freq/np.expand_dims(frame_totals,axis=1)

        self.freq = freq

        return freq


class Connectivity:
    '''
    Class for storing joint and linkage settings for dannce pose estimations
    '''

    def __init__(self, 
                 joint_names: Optional[List[str]]=[None], 
                 colors: Optional[List[Tuple[float,float,float,float]]]=[None], 
                 links: Optional[List[Tuple[int,int]]]=[None],
                 angles: Optional[List[Tuple[int,int,int]]]=[None]):

        self.joint_names=joint_names

        conn_dict = {'links':links,
                     'colors':colors}

        self.conn_df = pd.DataFrame(data=conn_dict)
        self.angles = angles

    @property
    def links(self):
        return list(self.conn_df['links'])

    @links.setter
    def links(self,
              links: List[Tuple[int,int]]):
        self.conn_df['links'] = links

    @property
    def colors(self):
        return list(self.conn_df['colors'])

    @colors.setter
    def colors(self,
               colors: List[Tuple[float,float,float,float]]):
        self.conn_df['colors'] = colors
        