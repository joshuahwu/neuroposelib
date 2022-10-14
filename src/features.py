import scipy as scp
# from scipy.ndimage import median_filter, convolve
from scipy.spatial.transform import Rotation as R
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import IncrementalPCA
from typing import Optional, Union, List, Tuple
import pandas as pd
from tqdm import tqdm
import h5py

def align_floor(pose: np.array,
                exp_id: Union[List, np.array],
                foot_id: Optional[int] = 12,
                head_id: Optional[int] = 0,
                plot_folder: Optional[str] = None):
    '''
        Due to calibration, predictions may be rotated on different axes
        Rotates floor to same x-y plane per video
        IN:
            pose: 3d matrix of (#frames x #joints x #coords)
            exp_id: Video ids per frame
            foot_id: ID of foot to find floor
        OUT:
            pose_rot: Floor aligned poses (#frames x #joints x #coords)
    '''
    pose_rot = pose
    print("Fitting and rotating the floor for each video to alignment ... ")
    for _,i in enumerate(tqdm(np.unique(exp_id))): # Separately for each video
        pose_exp = pose[exp_id == i,:,:]
        pose_exp = scp.ndimage.median_filter(pose_exp,(4,1,1)) # Median filter 5 frames repeat the ends of video

        # Initial calculation of plane to find outlier values
        [xy,z] = [pose_exp[:,foot_id,:2],pose_exp[:,foot_id,2]]
        const = np.ones((pose_exp.shape[0],1))
        coeff = np.linalg.lstsq(np.append(xy,const,axis=1),z,rcond=None)[0]
        z_diff = (pose_exp[:,foot_id,0]*coeff[0] + pose_exp[:,foot_id,1]*coeff[1] + coeff[2]) - pose_exp[:,foot_id,2]
        z_mean = np.mean(z_diff)
        z_range = np.std(z_diff)*1.5
        mid_foot_vals = np.where((z_diff>z_mean-z_range) & (z_diff<z_mean+z_range))[0] # Removing outlier values of foot

        # Recalculating plane with outlier values removed
        [xy,z] = [pose_exp[mid_foot_vals,foot_id,:2],pose_exp[mid_foot_vals,foot_id,2]]
        const = np.ones((xy.shape[0],1))
        coeff = np.linalg.lstsq(np.append(xy,const,axis=1),z,rcond=None)[0]
        
        # Calculating rotation matrices
        un = np.array([-coeff[0],-coeff[1],1])/np.linalg.norm([-coeff[0],-coeff[1],1])
        vn = np.array([0,0,1])
        theta = np.arccos(np.clip(np.dot(un,vn),-1,1))
        rot_vec = np.cross(un,vn)/np.linalg.norm(np.cross(un,vn))*theta
        rot_mat = R.from_rotvec(rot_vec).as_matrix()
        rot_mat = np.expand_dims(rot_mat, axis=2).repeat(pose_exp.shape[0]*pose_exp.shape[1],axis=2)
        pose_exp[:,:,2]-=coeff[2] # Fixing intercept to zero
        
        # Rotating
        pose_rot[exp_id==i,:,:] = np.einsum("jki,ik->ij", rot_mat, np.reshape(pose_exp,(-1,3))).reshape(pose_exp.shape)

        if plot_folder:
            xx, yy=np.meshgrid(range(-300,300,10),range(-300,300,10))
            zz = coeff[0]*xx + coeff[1]*yy + coeff[2]
            fig = plt.figure(figsize=(20,20))
            ax = plt.axes(projection='3d')
            # ax.scatter3D(pose_exp[1000,foot_id,0], pose_exp[1000,foot_id,1], pose_exp[1000,foot_id,2],s=1000,c='r')
            ax.scatter3D(pose_exp[:,foot_id,0], pose_exp[:,foot_id,1], pose_exp[:,foot_id,2],s=1,)
            ax.plot_surface(xx,yy,zz,alpha=0.2)
            plt.savefig(''.join([plot_folder,'/before_rot',str(i),'.png']))
            plt.close()

            fig = plt.figure(figsize=(20,20))
            ax = plt.axes()
            ax.scatter(pose_rot[exp_id==i,foot_id,0], pose_rot[exp_id==i,foot_id,2],s=1)
            # ax.scatter(pose_rot[1000,foot_id,0], pose_rot[1000,foot_id,2],s=100,c='r')
            plt.savefig('./after_rot',str(i),'.png')
            plt.close()

        ## Checking to make sure snout is on average above the feet
        assert(np.mean(pose_rot[exp_id==i,head_id,2])>np.mean(pose_rot[exp_id==i,foot_id,2])) #checking head is above foot

    return pose_rot

def center_spine(pose,
                 joint_idx = 4):
    print("Centering poses to mid spine ...")
    # Center spine_m to (0,0,0)
    return pose - np.expand_dims(pose[:,joint_idx,:],axis=1)

def rotate_spine(pose,
                 joint_idx = [4,3],
                 lock_to_x = False):
    '''
        Centers mid spine to (0,0,0) and aligns spine_m -> spine_f to x-z plane
        IN:
            pose: 3d matrix of (#frames x #joints x #coords)
            joint_idx: List [spine_m idx, spine_f idx]
            lock_to_x: Also rotates so that spine_m -> spine_f is locked to the x-axis
        OUT:
            pose_rot: Centered and rotated pose (#frames x #joints x #coords)
    '''
    num_joints = pose.shape[1]
    yaw = -np.arctan2(pose[:,joint_idx[1],1],pose[:,joint_idx[1],0]) # Find angle to rotate to axis

    if lock_to_x:
        print("Rotating spine to x axis ... ")
        pitch = -np.arctan2(pose[:,joint_idx[1],2],pose[:,joint_idx[1],0])
    else:
        print("Rotating spine to xz plane ... ")
        pitch = np.zeros(yaw.shape)

    # Rotation matrix for pitch and yaw
    rot_mat = np.array([[np.cos(yaw)*np.cos(pitch), -np.sin(yaw), np.cos(yaw)*np.sin(pitch)],
                        [np.sin(yaw)*np.cos(pitch), np.cos(yaw), np.sin(yaw)*np.sin(pitch)],
                        [-np.sin(pitch), np.zeros(len(yaw)), np.cos(pitch)]]).repeat(num_joints,axis=2)
    pose_rot = np.einsum("jki,ik->ij", rot_mat, np.reshape(pose,(-1,3))).reshape(pose.shape)

    # Making sure Y value of spine f doesn't deviate much from 0
    assert pose_rot[:,joint_idx[1],1].max()<1e-5 and pose_rot[:,joint_idx[1],1].min()>-1e-5
    if lock_to_x: # Making sure Z value of spine f doesn't deviate much from 0
        assert pose_rot[:,joint_idx[1],2].max()<1e-5 and pose_rot[:,joint_idx[1],2].min()>-1e-5

    return pose_rot

def get_lengths(pose,
                linkages):
    '''
        Get lengths of all linkages
    '''
    print("Calculating length of all linkages ... ")
    linkages = np.array(linkages)
    lengths = np.square(pose[:,linkages[:,1],:]-pose[:,linkages[:,0],:])
    lengths = np.sum(np.sqrt(lengths),axis=2)
    return lengths

def rolling_window(data, 
                   window):
    '''
        Returns a view of data windowed (data.shape, window)
        Pads the ends with the edge values
    '''
    try:
        assert(window%2 == 1)
    except ValueError:
        print("Window size must be odd")
        raise
    
    # Padding frames with the edge values with (window size/2 - 1)
    pad = int(np.floor(window/2))
    d_pad = np.pad(data,((pad, pad),(0,0)), mode='edge').T
    shape = d_pad.shape[:-1] + (d_pad.shape[-1] - pad*2, window)
    strides = d_pad.strides + (d_pad.strides[-1],)
    return np.swapaxes(np.lib.stride_tricks.as_strided(d_pad, shape=shape, strides=strides),0,1)

def get_velocities(pose,
                   exp_id,
                   joint_names,
                   joints=[0,3,5],
                   widths=[5,11,51],
                   sample_freq=90):
    '''
        Returns absolute velocity, as well as x, y, and z velocities over varying widths
        Also returns the standard deviation of these velocities over varying widths
        IN:
            pose: Non-centered and and optional rotated pose (#frames, #joints, #xyz)
            exp_id: Video ids per frame
            joints: joints to calculate absolute velocities
            widths: Number of frames to average velocity over (must be odd)
            sample_freq: Sampling frequency of the videos
        OUT:
            vel: velocity features (#frames x #joints*#widths)
    '''
    if np.any(np.sum(pose, axis=(0,2))==0):
        print("Detected centered pose input - calculating relative velocities ... ")
        tag = 'rel'
    else:
        print("Calculating absolute velocities ... ")
        tag = 'abs'

    ax_labels = ['vec','x','y','z']
    vel = np.zeros((pose.shape[0],len(joints)*len(widths)*len(ax_labels)))
    vel_stds = np.zeros(vel.shape)
    vel_labels, std_labels = [], []

    for _,i in enumerate(tqdm(np.unique(exp_id))): # Separating by video
        pose_exp = pose[exp_id==i,:,:][:,joints,:]

        # Calculate distance beetween  times t - (t-1)
        temp_pose = np.append(np.expand_dims(pose_exp[0,:,:],axis=0),pose_exp[:-1,:,:],axis=0)
        dxyz = np.reshape(pose_exp-temp_pose, (pose_exp.shape[0],-1)) # distance for each axis

        # Appending Euclidean vector magnitude of distance and multiplying by sample_freq to get final velocities
        dv = np.append(np.sqrt(np.sum(np.square(pose_exp-temp_pose),axis=2)),dxyz,axis=-1)*sample_freq

        # Calculate average velocity and velocity stds over the windows
        for j,width in enumerate(widths):
            kernel = np.ones((width,1))/width
            vel[exp_id==i,j*len(joints)*4:(j+1)*len(joints)*4] = scp.ndimage.convolve(dv, kernel, mode='constant')
            vel_stds[exp_id==j,j*len(joints)*4:(j+1)*len(joints)*4] = np.std(rolling_window(dv, width),axis=-1)

            if i==np.unique(exp_id)[0]:
                vel_labels+= ['_'.join([tag,'vel',ax,joint_names[joint],str(width)]) for joint in joints for ax in ax_labels]
                std_labels+= ['_'.join([tag,'vel_std',ax,joint_names[joint],str(width)]) for joint in joints for ax in ax_labels]
    
    # vel_feats = pd.DataFrame(np.hstack((vel,vel_stds)), columns=vel_labels+std_labels)
    return np.hstack((vel,vel_stds)), vel_labels+std_labels

def get_ego_pose(pose,
                 joint_names):
    '''
        Takes centered spine rotated pose - reshapes and converts to pandas dataframe
    '''
    print("Reformatting pose to egocentric pose features ... ")
    is_centered = np.any(np.sum(pose, axis=(0,2))==0)
    is_rotated = np.any(np.logical_and(np.sum(pose[:,:,1], axis=0)<1e-8, np.sum(pose[:,:,1], axis=0)>-1e-8))
    if not (is_centered and is_rotated):
        raise ValueError("Pose must be centered and rotated")

    pose = np.reshape(pose, (pose.shape[0],pose.shape[1]*pose.shape[2]))
    axis = ['x','y','z']
    labels = ['_'.join(['ego_euc',joint,ax]) for joint in joint_names for ax in axis]
    # pose_df = pd.DataFrame(pose,columns=labels)
    return pose, labels

def get_angles(pose,
               link_pairs):
    '''
        Calculates 3 angles for pairs of linkage vectors
        Angles calculated are those between projections of each vector onto the 3 xyz planes
        IN:
            pose: Centered and rotated pose (#frames, #joints, #xyz)
            link_pairs: List of tuples with 3 points between which to calculate angles
        OUT:
            angles: returns 3 angles between link pairs

        ** Currently doing unsigned
    '''
    print("Calculating joint angles ... ")
    angles = np.zeros((pose.shape[0],len(link_pairs),3))
    feat_labels = []
    plane_dict = {'xy':[0,1],
                  'xz':[0,2],
                  'yz':[1,2]}
    for i,pair in enumerate(tqdm(link_pairs)):
        v1 = pose[:,pair[0],:]-pose[:,pair[1],:] #Calculate vectors
        v2 = pose[:,pair[2],:]-pose[:,pair[1],:]
        for j,key in enumerate(plane_dict):
            # This is for signed angle
            # angles[:,i,j] = np.arctan2(v1[:,plane_dict[key][0]],v1[:,plane_dict[key][1]]) - \
            #                 np.arctan2(v2[:,plane_dict[key][0]],v2[:,plane_dict[key][1]])

            # This is for unsigned angle
            v1_u = v1[:,plane_dict[key]]/np.expand_dims(np.linalg.norm(v1[:,plane_dict[key]],axis=1),axis=1)
            v2_u = v2[:,plane_dict[key]]/np.expand_dims(np.linalg.norm(v2[:,plane_dict[key]],axis=1),axis=1)
            angles[:,i,j] = np.arccos(np.clip(np.sum(v1_u*v2_u,axis=1),-1, 1))

            feat_labels += ['_'.join(['ang'] + [str(i) for i in pair] + [key])]

    # Fix all negative angles so that final is between 0 and 2pi
    # round_offset = 1e-4
    # angles = np.clip(angles, -2*np.pi+round_offset, 2*np.pi-round_offset)
    # angles = np.where(angles>0, angles, angles+2*np.pi)

    angles = np.reshape(angles,(angles.shape[0],angles.shape[1]*angles.shape[2]))
    # angles = pd.DataFrame(angles, columns=feat_labels)
    return angles, feat_labels

def get_angular_vel(angles,
                    angle_labels,
                    exp_id,
                    widths=[5,11,51],
                    sample_freq = 90):
    '''
        Calculates angular velocity of previously defined angles
        IN:
            angles: Pandas dataframe of angles ()
    '''
    print("Calculating velocities of angles ... ")
    num_ang = angles.shape[1]
    avel = np.zeros((angles.shape[0],num_ang*len(widths)))
    avel_stds = np.zeros(avel.shape)
    avel_labels, std_labels = [], []
    for _,i in enumerate(tqdm(np.unique(exp_id))):
        ang_exp = angles[exp_id==i,:]
        prev_ang = np.append(np.expand_dims(ang_exp[0,:],axis=0),ang_exp[:-1,:],axis=0)
        dtheta = (ang_exp - prev_ang)*sample_freq
        for j,width in enumerate(widths):
            kernel = np.ones((width,1))/width
            avel[exp_id==i,j*num_ang:(j+1)*num_ang] = scp.ndimage.convolve(dtheta, kernel, mode='constant')
            avel_stds[exp_id==j,j*num_ang:(j+1)*num_ang] = np.std(rolling_window(dtheta, width),axis=-1)

            if i == np.unique(exp_id)[0]:
                avel_labels+=['_'.join([label.replace("ang","avel"),str(width)]) for label in angle_labels]
                std_labels+=['_'.join([label.replace("ang","avel_std"),str(width)]) for label in angle_labels]

    # avel_feats = pd.DataFrame(np.hstack((avel,avel_stds)), columns=avel_labels+std_labels)

    return np.hstack((avel,avel_stds)), avel_labels+std_labels

def get_head_angular(pose,
                     exp_id,
                     widths=[5,10,50],
                     link = [0,3,4]):
    '''
        Getting x-y angular velocity of head
        IN:
            pose: Non-centered, optional rotated pose
    '''
    v1 = pose[:,link[0],:2]-pose[:,link[1],:2]
    v2 = pose[:,link[2],:2]-pose[:,link[1],:2]

    angle= np.arctan2(v1[:,0],v1[:,1]) - np.arctan2(v2[:,0],v2[:,1])
    angle = np.where(angle>0, angle, angle+2*np.pi)

    angular_vel = np.zeros((len(angle),len(widths)))
    for _,i in tqdm.tqdm(np.unique(exp_id)):
        angle_exp = angle[exp_id==i]
        d_angv = angle_exp - np.append(angle_exp[0],angle_exp[:-1])
        for i,width in enumerate(widths):
            kernel = np.ones(width)/width
            angular_vel[exp_id==i,i] = scp.ndimage.convolve(d_angv, kernel, mode='constant')

    return angular_vel

def wavelet(features,
            labels,
            exp_id,
            sample_freq = 90,
            freq = np.linspace(1,25,25),
            w0 = 5):
    # scp.signal.morlet2(500, )
    print("Calculating wavelets ... ")
    widths = w0*sample_freq/(2*freq*np.pi)
    wlet_feats = np.zeros((features.shape[0],len(freq)*features.shape[1]))

    wlet_labels = ['_'.join(['wlet',label,str(f)]) for label in labels for f in freq]

    for i in np.unique(exp_id):
        print("Calculating wavelets for video " + str(i))
        for j in tqdm(range(features.shape[1])):
            wlet_feats[exp_id==i,j*len(freq):(j+1)*len(freq)] = np.abs(scp.signal.cwt(features[exp_id==i,j],
                                                                                      scp.signal.morlet2, 
                                                                                      widths,
                                                                                      w=w0).T)
    return wlet_feats, wlet_labels

def pca(features,
        labels,
        categories = ['vel','ego_euc','ang','avel'],
        n_pcs = 10,
        method = 'ipca'):
    print("Calculating principal components ... ")
    import time
    # Initializing the PCA method
    if method.startswith('torch'):
        import torch
        pca_feats = torch.zeros(features.shape[0], len(categories)*n_pcs)
        features = torch.tensor(features)
    else:
        # Centering the features if not torch (pytorch does it itself)
        features = features - features.mean(axis=0)
        pca_feats = np.zeros((features.shape[0], len(categories)*n_pcs))

    if method == 'ipca':
        from sklearn.decomposition import IncrementalPCA
        pca = IncrementalPCA(n_components=n_pcs, batch_size=None)
    elif method.startswith('fbpca'):
        import fbpca

    num_cols = 0
    for i, cat in enumerate(tqdm(categories)): # Iterate through each feature category
        cat += '_'
        cols_idx = [i for i, col in enumerate(labels) if (col.startswith(cat) or ('_'+cat in col))]
        num_cols += len(cols_idx)
        
        if method=='ipca' or method=='sklearn_pca':
            # import pdb; pdb.set_trace()
            pca_feats[:, i*n_pcs:(i+1)*n_pcs] = pca.fit_transform(features[:,cols_idx])

        elif method.startswith('torch'):
            feat_cat = features[:,cols_idx]
            if method.endswith('_gpu'):
                feat_cat = feat_cat.cuda()

            if 'pca' in method:
                (_,_,V) = torch.pca_lowrank(feat_cat)
            elif 'svd' in method:
                feat_cat -= feat_cat.mean()
                (_,_,V) = torch.linalg.svd(feat_cat)

            if method.endswith('_gpu'):
                pca_feats[:, i*n_pcs:(i+1)*n_pcs] = torch.matmul(feat_cat, V[:,:n_pcs]).detach().cpu()
                feat_cat.detach().cpu()
                V.detach().cpu()
            else:
                pca_feats[:, i*n_pcs:(i+1)*n_pcs] = torch.matmul(feat_cat, V[:,:n_pcs])

        elif method == 'fbpca':
            (_,_,V) = fbpca.pca(features[:, cols_idx], k = n_pcs)
            pca_feats[:, i*n_pcs:(i+1)*n_pcs] = np.matmul(features[:, cols_idx], V.T)

    if method.startswith('torch_pca'):
        pca_feats = pca_feats.numpy()
    
    assert num_cols == features.shape[1]

    pc_labels = ['_'.join([cat,'pc'+str(i)]) for cat in categories for i in range(n_pcs)]

    return pca_feats, pc_labels

def save_h5(features,
            labels,
            path):
    '''
        Writes to h5 file
    '''
    hf = h5py.File(path, 'w')
    hf.create_dataset('features', data=features)
    str_dtype = h5py.special_dtype(vlen=str)
    hf.create_dataset('labels', data=labels, dtype=str_dtype)
    hf.close()
    return

def read_h5(path):
    '''
        Reads h5 file
    '''
    hf = h5py.File(path,'r')
    features = np.array(hf.get('features'))
    labels = np.array(hf.get('labels'), dtype=str).tolist()
    hf.close()
    return features, labels