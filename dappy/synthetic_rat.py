import numpy as np
import read
import DataStruct as ds
import features
import run
import visualization as vis
import pandas as pd
from pathlib import Path
import pickle

analysis_key = 'synthetic_rat'
paths = read.config("../configs/path_configs/" + analysis_key + ".yaml")
params = read.config("../configs/param_configs/fitsne.yaml")

if not Path(paths['out_path'] + params["label"]+'/datastruct.p').exists():
    data_obj = pickle.load(
        open("".join([paths["out_path"], params["label"], "/datastruct.p"]), "rb")
    )
else:
    connectivity = read.connectivity(
            path=paths["skeleton_path"], skeleton_name=paths["skeleton_name"]
        )

    action_id_to_action = {
        0: 'smallSniffZ',
        1: 'headZ',
        2: 'groomZ',
        3:'rearedSlowZ',
        4: 'scrunchedZ',
        5: 'crouchedZ',
        6: 'crouchedFastZ',
        7: 'rearedZ',
        8: 'groundExpZ',
        9: 'exploreZ',
        10: 'stepsExploreZ',
        11: 'locZ',
        12: 'fastLocZ'
    }
    action_to_action_id = {v: k for k, v in action_id_to_action.items()}

    pose = np.load(paths['data_path']+'pose.npy') # 500 samples x 13 actions x 23 keypoints x 3 coordinates x 60 frames
    ids = np.tile(np.arange(pose.shape[0]*pose.shape[1])[:,None], pose.shape[-1]).flatten()

    num_frames_per_class = pose.shape[0]*pose.shape[1]*pose.shape[-1]
    num_class = len(action_id_to_action.keys())

    # action_id = np.tile(np.arange(num_class)[:,None],num_frames_per_class).flatten()
    meta = pd.DataFrame()
    meta['id'] = np.arange(pose.shape[0]*pose.shape[1])
    meta['action'] = np.repeat(np.arange(num_class),pose.shape[0])
    meta_by_frame = meta.iloc[ids].reset_index(drop=True)

    pose_reshaped = np.moveaxis(np.moveaxis(pose,4,2),1,0).reshape((-1,23,3))
    assert np.all(np.linalg.norm(pose_reshaped[:60,...]-np.moveaxis(pose[0,0,...],-1,0))<1e-5)

    pose = features.rotate_spine(features.center_spine(pose_reshaped))

    features, labels = run.standard_features(pose, connectivity, paths)

    data_obj = run.run(
            features,
            labels,
            pose,
            ids,
            connectivity,
            paths,
            params,
            meta,
            meta_by_frame
        )

    data_obj.write_pickle("".join([paths["out_path"], params["label"], "/"]))

for action in range(13):
    action_bool = data_obj.data["action"].values.astype(int) == action
    sizes = [20 if is_action else 3 for is_action in action_bool]
    vis.scatter_by_cat(data_obj.embed_vals, 1*action_bool, color = [(0.5,0.5,0.5), (1, 0.5, 0)], size = sizes,
                       label='action',filepath = ''.join([paths['out_path'],params['label'],'/action/',str(action),'_']))

import pdb; pdb.set_trace()