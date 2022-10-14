import pandas as pd
import numpy as np
import h5py
import hdf5storage
import matplotlib.pyplot as plt
from typing import Optional, Union, List, Tuple
import yaml

class DataStruct:

    _props = [
        'analysis_path',
        'pose_path',
        'meta_path',
        'out_path',
        'exp_key',
        'embed_vals',
        'data',
        'meta_by_frame',
        'frame_id',
        'features',
        'meta',
        'downsample',
    ]

    def __init__(self,
                 data: pd.DataFrame = pd.DataFrame(),
                 meta: pd.DataFrame = pd.DataFrame(),
                 full_data: pd.DataFrame = pd.DataFrame(),
                 connectivity = None,
                 out_path: Optional[str] = None,
                 downsample: Optional[int] = None,
                 config_path: Optional[str] = None):

        self.config_path = config_path
        
        (
            self.analysis_path,
            self.pose_path,
            self.meta_path,
            self.out_path,
            self.skeleton_path,
            self.skeleton_name,
            self.exp_key,
        ) = tuple(self.read_config(config_path).values())

        if out_path is not None:
            self.out_path = out_path
            
        self.downsample = downsample
        self.data = data
        self.meta = meta
        self.full_data = full_data
        self.connectivity = connectivity
        self.exp_ids_full = None

    def __getitem__(self, idx):
        if not isinstance(idx, tuple):
            idx=tuple(idx)
        new_struct = DataStruct(data = self.data.loc[idx].reset_index(drop=True),
                                meta = self.meta,
                                full_data = self.full_data,
                                connectivity = self.connectivity,
                                out_path = self.out_path,
                                downsample = self.downsample,
                                config_path = self.config_path)
        return new_struct
        
    def check_reset_data(self, len:int):
        if self.data.shape[0] != len:
            self.data = pd.DataFrame()

    @property
    def frame_id(self):
        return self.data['frame_id'].to_numpy()

    @frame_id.setter
    def frame_id(self,
                 frame_id: Union[List[int],np.array]):
        self.data['frame_id'] = frame_id
    
    @property
    def exp_id(self):
        return self.data['exp_id'].to_numpy()

    @exp_id.setter
    def exp_id(self,
               exp_id: Union[List[Union[str, int]],np.array]):
        self.data['exp_id'] = exp_id

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
        if out_path is None:
            out_path = self.out_path
        import pickle
        pickle.dump(self, open(''.join([out_path,"datastruct.p"]), "wb"))

    def cluster_freq(self):
        '''
        Calculates the percentage of time each exp_id spends in each cluster
        OUT:
            freq: numpy array of (# videos x # clusters)
        '''
        num_clusters = np.max(self.data['Cluster'])+1
        freq = np.zeros((len(self.meta.index), num_clusters))
        for i in self.meta.index:
            cluster_by_exp = self.data['Cluster'][self.data['exp_id']==i]
            freq[i,:] = np.histogram(cluster_by_exp, bins=num_clusters, range=(-0.5,num_clusters-0.5))[0]
        frame_totals = np.sum(freq,axis=1)
        frame_totals = np.where(frame_totals==0, 1, frame_totals)
        freq = freq/np.expand_dims(frame_totals,axis=1)

        self.freq = freq

        return freq

    def read_config(self,
                    filepath,
                    config_params: Optional[List[str]] = None):
        '''
        Read configuration file and set instance attributes 
        based on key, value pairs in the config file

        IN:
            filepath - Path to configuration file
        OUT:
            config_dict - Dict of path variables to data in config file
        '''
        if config_params is None:
            config_params = ['analysis_path','pose_path','meta_path','out_path',
                             'skeleton_path','skeleton_name',
                             'exp_key']

        with open(filepath) as f:
            config_dict = yaml.safe_load(f)

        for key in config_params:
            if key not in config_dict:
                config_dict[key]=None

        return config_dict
        
    def load_meta(self, 
                  meta_path: Optional[str] = None,
                  exp_id: Optional[List[Union[str, int]]]=None,
                  return_out: bool = False):
        if meta_path: self.meta_path = meta_path
        if exp_id: self.exp_id = exp_id

        meta = pd.read_csv(self.meta_path)
        meta_by_frame = meta.iloc[self.exp_id].reset_index().rename(columns={'index':'exp_id'})

        self.meta = meta
        self.meta_by_frame = meta_by_frame

        if return_out:
            return meta, meta_by_frame

        return self

    def load_feats(self,
                   analysis_path: Optional[str]=None, 
                   pose_path: Optional[str]=None, 
                   exp_key: Optional[str]=None, 
                   downsample: int = 20, 
                   return_out: bool = False):
        '''
        Load in data (we only care about exp_id, frames_with_good_tracking and jt_features)

        IN:
            analysis_path - Path to MATLAB analysis struct with jt_features included
            pose_path - Path to predictions .mat file
            exp_key - Name of category to separate by experiment
            downsample - Factor by which to downsample features and IDs for analysis

        OUT:
            features - Numpy array of features for each frame for analysis (frames x features)
            exp_id - List of labels for categories based on the exp_key
            frames_with_good_tracking - Indices in merged predictions file to keep track of downsampling
        '''
        if analysis_path: self.analysis_path = analysis_path
        if pose_path: self.pose_path = pose_path
        if exp_key: self.exp_key = exp_key

        analysisstruct = hdf5storage.loadmat(self.analysis_path, 
                                             variable_names=['jt_features',
                                                             'frames_with_good_tracking',
                                                             'tsnegranularity'])
        features = analysisstruct['jt_features']

        try:
            frames_with_good_tracking = np.squeeze(analysisstruct['frames_with_good_tracking'][0][0].astype(int))-1
        except:
            frames_with_good_tracking = np.squeeze(analysisstruct['frames_with_good_tracking'][0][1].astype(int))-1

        if self.exp_ids_full is None:
            exp_ids_full = np.squeeze(hdf5storage.loadmat(self.pose_path, variable_names=[self.exp_key])[self.exp_key].astype(int))

            if np.min(exp_ids_full)!=0:
                exp_ids_full -= np.min(exp_ids_full)

            self.exp_ids_full = exp_ids_full

        exp_id = self.exp_ids_full[frames_with_good_tracking] # Indexing out batch IDs

        print("Size of dataset: ", np.shape(features))

        # downsample
        frames_with_good_tracking = frames_with_good_tracking[::downsample]
        features = features[::downsample]
        exp_id = exp_id[::downsample]

        self.exp_id = exp_id
        self.frame_id = frames_with_good_tracking
        self.features = features
        self.downsample = downsample*int(analysisstruct['tsnegranularity'])

        if return_out:
            return features, exp_id, frames_with_good_tracking

        return self

    def load_pose(self, 
                  pose_path: Optional[str] = None,
                  connectivity = None,
                  return_out: bool = False):

        if pose_path: self.pose_path = pose_path
        if connectivity: self.connectivity = connectivity

        try:
            f = h5py.File(self.pose_path)['markers_preproc_rotated']
            mat_v7 = True
            total_frames = max(np.shape(f[list(f.keys())[0]]))
        except:
            print("Detected older version of '.mat' file")
            f = hdf5storage.loadmat(self.pose_path, variable_names=['predictions'])['predictions']
            mat_v7 = False
            total_frames = max(np.shape(f[0][0][0]))

            if self.exp_ids_full is None:
                exp_ids_full = np.squeeze(hdf5storage.loadmat(self.pose_path, variable_names=[self.exp_key])[self.exp_key].astype(int))

                if np.min(exp_ids_full)!=0:
                    exp_ids_full -= np.min(exp_ids_full)

                self.exp_ids_full = exp_ids_full
            

        pose_3d = np.empty((total_frames, 0, 3))
        for key in self.connectivity.joint_names:
            print(key)
            try:
                if mat_v7:
                    joint_preds = np.expand_dims(np.array(f[key]).T,axis=1)
                else:
                    joint_preds = np.expand_dims(f[key][0][0],axis=1)
            except:
                print("Could not find ",key," in preds")
                continue
            
            pose_3d = np.append(pose_3d, joint_preds, axis=1)
        
        self.pose_3d = pose_3d
        if return_out:
            return pose_3d

        return self

    def load_connectivity(self, 
                          skeleton_path: Optional[str] = None, 
                          skeleton_name: Optional[str] = None,
                          return_out: bool = False):

        if skeleton_path: self.skeleton_path=skeleton_path
        if skeleton_name: self.skeleton_name=skeleton_name

        self.connectivity = Connectivity().load(skeleton_path = self.skeleton_path,
                                                skeleton_name = self.skeleton_name)

        if return_out:
            return self.connectivity

        return self

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

    def load(self, 
             skeleton_path: str, 
             skeleton_name: str = 'mouse20'):
        '''
        Load in joint names, connectivities and colors for connectivites
        IN:
            skeleton_path: Path to Python file with dicts for skeleton information
            skeleton_name: Key for correct skeleton in connectivity dict

        '''
        if skeleton_path.endswith('.py'):
            import importlib.util
            mod_spec = importlib.util.spec_from_file_location('connectivity',skeleton_path)
            con = importlib.util.module_from_spec(mod_spec)
            mod_spec.loader.exec_module(con)

            self.conn_df = pd.DataFrame()
            self.joint_names = con.JOINT_NAME_DICT[skeleton_name] # joint names
            self.colors = con.COLOR_DICT[skeleton_name] # color to be plotted for each linkage
            self.links = con.CONNECTIVITY_DICT[skeleton_name] # joint linkages
            self.angles = con.JOINT_ANGLES_DICT[skeleton_name] # angles to calculate

        return self
        