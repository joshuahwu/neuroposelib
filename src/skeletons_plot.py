import pickle
import visualization as vis
from DataStruct import DataStruct

path = '../results/24_final/fitsne_425/datastruct.p'

ds = pickle.load(open(path,'rb'))
# vis.scatter_on_watershed(data=ds, watershed=ds.ws, column='Cluster')

# ds.pose_path = '/home/exx/Desktop/GitHub/CAPTURE_demo/CAPTURE_data/48_tadross_data/markers_preproc_rotated.mat'
# ds.load_connectivity()
# ds.load_pose()
# vis.skeleton_vid3D_cat(ds, 'Cluster', n_skeletons=9)

new_struct = DataStruct(data = ds.data,
                        meta = ds.meta,
                        full_data = ds.full_data,
                        connectivity = ds.connectivity,
                        out_path = ds.out_path,
                        downsample = ds.downsample,
                        config_path = ds.config_path)

new_struct.ws = ds.ws

# vis.density(ds.ws.density, ds.ws.borders,
#                 filepath = ''.join([new_struct.out_path,'final_density.png']))
# vis.scatter(new_struct.embed_vals, filepath=''.join([new_struct.out_path,'final_scatter.png']))
# column = ['Condition']
# for cat in column:
#     vis.density_cat(data=new_struct, column=cat, watershed=new_struct.ws, n_col=12,
#                     filepath = ''.join([new_struct.out_path,'final_density_',cat,'.png']))

new_struct.cluster_freq()