import h5py
import numpy as np

def features_h5(features,
                labels,
                path):
    '''
        Writes features and labels to h5 file
    '''
    hf = h5py.File(path, 'w')
    hf.create_dataset('features', data=features)
    str_dtype = h5py.special_dtype(vlen=str)
    hf.create_dataset('labels', data=labels, dtype=str_dtype)
    hf.close()
    return

def pose_h5(pose,
            exp_ids,
            exp_key,
            path):
    '''
        Writes poses (#frames x #joints x `xyz` to h5 file)
    '''
    hf = h5py.File(path, 'w')
    hf.create_dataset('pose', data=pose)
    hf.create_dataset(exp_key, data=exp_ids)
    hf.close()
    return