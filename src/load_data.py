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

    exp_ids_full = np.squeeze(hdf5storage.loadmat(self.pose_path, variable_names=[self.exp_key])[self.exp_key].astype(int))

    if np.min(exp_ids_full)!=0:
        exp_ids_full -= np.min(exp_ids_full)

    self.exp_ids_full = exp_ids_full

    exp_id = exp_ids_full[frames_with_good_tracking] # Indexing out batch IDs

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