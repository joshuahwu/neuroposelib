from neuroposelib.visualization.projection_api import *
from tqdm import tqdm
import matplotlib.colors as mcolors
from pathlib import Path


def get_brights_list():
    bright_colors = {
                        name: color
                        for name, color in mcolors.XKCD_COLORS.items()
                        if any(phrase in name for phrase in ["bright", "neon", "vivid", "hot"])
                    }



    # Convert to RGBA with alpha=0.7
    col_names = list(mcolors.XKCD_COLORS.keys())
    col_names.sort()
    # rgba_colors = [mcolors.to_rgba(bright_colors[name], alpha=0.7) for name in col_names]
    # rgba_colors.sort()

    rgba_colors = [mcolors.to_rgba('xkcd:purple', alpha=0.7),
                   mcolors.to_rgba('xkcd:red', alpha=0.7),
                   mcolors.to_rgba('xkcd:blue', alpha=0.7),
                   mcolors.to_rgba('xkcd:brown', alpha=0.7),
                   mcolors.to_rgba('xkcd:pink', alpha=0.7),
                   mcolors.to_rgba('xkcd:orange', alpha=0.7),
                   mcolors.to_rgba('xkcd:sky blue', alpha=0.7),
                   mcolors.to_rgba('xkcd:royal blue', alpha=0.7),
                   mcolors.to_rgba('xkcd:salmon', alpha=0.7),
                   mcolors.to_rgba('xkcd:bright yellow', alpha=0.7),
                   mcolors.to_rgba('xkcd:neon blue', alpha=0.7),
                   mcolors.to_rgba('xkcd:neon red', alpha=0.7),
                   mcolors.to_rgba('xkcd:leather', alpha=0.7),
                   mcolors.to_rgba('xkcd:rust brown', alpha=0.7),
                   mcolors.to_rgba('xkcd:tealish green', alpha=0.7),
                   mcolors.to_rgba('xkcd:battleship grey', alpha=0.7),
                   mcolors.to_rgba('xkcd:off blue', alpha=0.7),
                   mcolors.to_rgba('xkcd:macaroni and cheese', alpha=0.7),
                   mcolors.to_rgba('xkcd:light eggplant', alpha=0.7),
                   mcolors.to_rgba('xkcd:dark green blue', alpha=0.7),
                   mcolors.to_rgba('xkcd:bright sea green', alpha=0.7),
                   mcolors.to_rgba('xkcd:strong blue', alpha=0.7),
                   mcolors.to_rgba('xkcd:indian red', alpha=0.7),
                   mcolors.to_rgba('xkcd:egg shell', alpha=0.7),
                   mcolors.to_rgba('xkcd:custard', alpha=0.7),
                   mcolors.to_rgba('xkcd:topaz', alpha=0.7),
                   ]

    return rgba_colors

def crop_frame(frame, x1, y1, x2, y2):
    # Check if any dimensions overflow:
    # print(x1,x2,y1,y2)
    # print(frame.shape)
    if y2 > frame.shape[0]:
        y2_ = y2
        y2 = frame.shape[0]
        y1 = frame.shape[0] - (y2_-y1)
    if x2 > frame.shape[1]:
        x2_ = x2
        x2 = frame.shape[1]
        x1 = frame.shape[1] - (x2_-x1)
    if y1 < 0:
        y1_ = y1
        y1 = 0
        y2 = y1 + (y2-y1_)
    if x1 < 0:
        x1_ = x1
        x1 = 0
        x2 = x1 + (x2-x1_)
    # print(x1,x2,y1,y2)
    # return frame[x1:x2, y1:y2]
    return x1,x2,y1,y2

def find_nearest_bigger_block_size(init_block_size):

    for i in range(16):
        print ((init_block_size + i) % 16)
        if (init_block_size + i) % 16 == 0:
            break
    # print("Init block size = ", init_block_size)
    # print("MAX square side size = ", init_block_size + i)
    return init_block_size + i

def get_max_sides_per_cam(predictions_2d):
    head_KPs = predictions_2d[:,:,:5,:]
    # max_1 = np.max(np.linalg.norm(head_KPs[:,:,0,:] - head_KPs[:,:,1,:], axis=2), axis=1)
    # max_2 = np.max(np.linalg.norm(head_KPs[:,:,0,:] - head_KPs[:,:,2,:], axis=2), axis=1)
    # max_3 = np.max(np.linalg.norm(head_KPs[:,:,1,:] - head_KPs[:,:,2,:], axis=2), axis=1)
    max_4 = np.max(np.linalg.norm(head_KPs[:,:,0,:] - head_KPs[:,:,3,:], axis=2), axis=1)
    scaling_factor = 80/max_4[0]
    # return [find_nearest_bigger_block_size(i) for i in (np.max(np.vstack([max_1, max_2, max_3]), axis=0) + 10).astype(int)]
    # return find_nearest_bigger_block_size(np.max([max_1, max_2, max_3]).astype(int))
    # return find_nearest_bigger_block_size(np.max(max_4).astype(int))
    return [find_nearest_bigger_block_size(int(i//2 + i*scaling_factor)*1.5) for i in max_4]

def crop_vid_dbg(predictions_2d,
             viddir_path,
             square_side,
             com_list,
             params,
             sync,
             predictions,
             start_sample = 0,
             max_samples=1000
             ):

    # Input video file and output video file
        
    for ncam in range (len(params)):

        input_video_path = os.path.join(viddir_path, 'Camera{}/'.format(ncam+1)+'0.mp4')
        output_video_path = os.path.join(viddir_path, 'Camera{}/'.format(ncam+1)+'cropped_new_0.mp4')

        # Create a reader and writer
        reader = imageio.get_reader(input_video_path)
        writer = imageio.get_writer(output_video_path, fps=reader.get_meta_data()['fps'])

        # Get the max side per cam
        max_side = square_side[ncam]

        fig, axes = plt.subplots(1, 2, figsize=(8, 8), dpi=300)
        print("="*40 + "Starting for camera {}".format(ncam+1)+ "="*40)
        # Iterate through frames, crop, and write to the new video
        for frame_number in tqdm(range(start_sample, start_sample + max_samples)):
            fr = sync[0]["data_frame"][(np.where(sync[0]["data_sampleID"] == predictions["sampleID"][0][frame_number]))[0].squeeze()]
            frame = reader.get_data(fr[0])

            # Get the head_com in the frame
            head_COM = com_list[ncam][frame_number]
            # Crop region coordinate
            (x1, x2, y1, y2) = (int(head_COM[0])-int(max_side),
                                int(head_COM[0])+int(max_side), 
                                int(head_COM[1])-int(max_side),
                                int(head_COM[1])+int(max_side)) 
            # (x1, x2, y1, y2) = ((frame.shape[0] - int(head_COM[0]))-int(max_side//2),
            #                     (frame.shape[0] - int(head_COM[0]))+int(max_side//2), 
            #                     (frame.shape[1] - int(head_COM[1]))-int(max_side//2),
            #                     (frame.shape[1] - int(head_COM[1]))+int(max_side//2))
            # import pdb; pdb.set_trace()
            
            
            # for frame_number, frame in enumerate(reader):
            nx1, nx2, ny1, ny2 = crop_frame(frame, x1, y1, x2, y2)
            cropped_frame = frame
            axes[0].imshow(frame)
            axes[0].scatter(head_COM[0], head_COM[1], marker='.', color='red', linewidths=1)
            axes[0].scatter(nx1,ny1, c = 'g')
            axes[0].scatter(nx1,ny2, c = 'g')
            axes[0].scatter(nx2,ny1, c = 'g')
            axes[0].scatter(nx2,ny2, c = 'g')
            axes[0].plot((x1,x1),(y1,y2), c='r', alpha=0.7)
            axes[0].plot((x1,x2),(y2,y2), c='r', alpha=0.7)
            axes[0].plot((x2,x2),(y2,y1), c='r', alpha=0.7)
            axes[0].plot((x2,x1),(y1,y1), c='r', alpha=0.7)
            axes[1].imshow(frame[ny1:ny2, nx1:nx2, :])
            # axes.plot((x2,y1),(x2,y2), c='r', alpha=0.7)
            fig.canvas.draw()
            cropped_frame = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            cropped_frame = cropped_frame.reshape(fig.canvas.get_width_height()[::-1] + (3,))

            if cropped_frame.shape[0] != cropped_frame.shape[1]:
                print('x1 {}, y1 {}, x2 {}, y2 {}'.format(x1, y1, x2, y2))
                print("Image size = {}, {}".format(y2-y1, x2-x1))
                print("Cropped Frame Size = {}".format(cropped_frame.shape))
            
            writer.append_data(cropped_frame)
            axes[0].clear()
            axes[1].clear()

        # Close the video writer
        writer.close()


def crop_vid(predictions_2d,
             viddir_path,
             square_side,
             com_list,
             params,
             sync,
             predictions,
             start_sample = 0,
             max_samples=1000,
             new_vid_name="cropped_new"
             ):

    # Input video file and output video file
        
    for ncam in range (len(params)):

        input_video_path = os.path.join(viddir_path, 'Camera{}/'.format(ncam+1)+'0.mp4')
        output_video_path = os.path.join(viddir_path, 'Camera{}/'.format(ncam+1)+new_vid_name+'_0.mp4')

        # Create a reader and writer
        reader = imageio.get_reader(input_video_path)
        writer = imageio.get_writer(output_video_path, fps=reader.get_meta_data()['fps'])
        print("Current File FPS = {}".format(reader.get_meta_data()['fps']))
        # Get the max side per cam
        max_side = square_side[ncam]

        fig, axes = plt.subplots(1, 1, figsize=(8, 8), dpi=300)
        print("="*40 + "Starting for camera {}".format(ncam+1)+ "="*40)
        # Iterate through frames, crop, and write to the new video
        for frame_number in tqdm(range(start_sample, start_sample + max_samples)):
            fr = sync[0]["data_frame"][(np.where(sync[0]["data_sampleID"] == predictions["sampleID"][0][frame_number]))[0].squeeze()]
            frame = reader.get_data(fr[0])

            # Get the head_com in the frame
            head_COM = com_list[ncam][frame_number]
            # Crop region coordinate
            (x1, x2, y1, y2) = (int(head_COM[0])-int(max_side),
                                int(head_COM[0])+int(max_side), 
                                int(head_COM[1])-int(max_side),
                                int(head_COM[1])+int(max_side)) 
            # Crop to only snout, whisker and eye
            # (x1, x2, y1, y2) = ((frame.shape[0] - int(head_COM[0]))-int(max_side//2),
            #                     (frame.shape[0] - int(head_COM[0]))+int(max_side//2), 
            #                     (frame.shape[1] - int(head_COM[1]))-int(max_side//2),
            #                     (frame.shape[1] - int(head_COM[1]))+int(max_side//2))
            # import pdb; pdb.set_trace()
            
            
            # for frame_number, frame in enumerate(reader):
            nx1, nx2, ny1, ny2 = crop_frame(frame, x1, y1, x2, y2)
            # cropped_frame = frame
            # axes[0].imshow(frame)
            # axes[0].scatter(head_COM[0], head_COM[1], marker='.', color='red', linewidths=1)
            # axes[0].scatter(nx1,ny1, c = 'g')
            # axes[0].scatter(nx1,ny2, c = 'g')
            # axes[0].scatter(nx2,ny1, c = 'g')
            # axes[0].scatter(nx2,ny2, c = 'g')
            # axes[0].plot((x1,x1),(y1,y2), c='r', alpha=0.7)
            # axes[0].plot((x1,x2),(y2,y2), c='r', alpha=0.7)
            # axes[0].plot((x2,x2),(y2,y1), c='r', alpha=0.7)
            # axes[0].plot((x2,x1),(y1,y1), c='r', alpha=0.7)
            # axes[1].imshow(frame[ny1:ny2, nx1:nx2, :])
            # # axes.plot((x2,y1),(x2,y2), c='r', alpha=0.7)
            # fig.canvas.draw()
            # cropped_frame = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            # cropped_frame = cropped_frame.reshape(fig.canvas.get_width_height()[::-1] + (3,))

            cropped_frame = frame[ny1:ny2, nx1:nx2, :]

            if cropped_frame.shape[0] != cropped_frame.shape[1]:
                print('x1 {}, y1 {}, x2 {}, y2 {}'.format(x1, y1, x2, y2))
                print("Image size = {}, {}".format(y2-y1, x2-x1))
                print("Cropped Frame Size = {}".format(cropped_frame.shape))
            
            writer.append_data(cropped_frame)
            # axes[0].clear()
            # axes[1].clear()

        # Close the video writer
        writer.close()


def full_bod_vid(predictions_2d,
             viddir_path,
             com_list,
             params,
             sync,
             predictions,
             links,
             goodmarks,
             color_dict,
             joint_names=None,
             start_sample = 0,
             max_samples=1000,
             new_vid_name="cropped_new",
             fps=None,
             vid_id = 3000,
             annot = True,
             vid_out_path=None,
             ):

    # Input video file and output video file
    bright_colors_list = get_brights_list()
        
    for ncam in range (len(params)):

        input_video_path = os.path.join(viddir_path, 'Camera{}/'.format(ncam+1)+'{}.mp4'.format(vid_id))
        if vid_out_path:
            # save_dir_path = save_path.joinpath(f'cluster_{idx+1}')
            # try:
            #     save_dir_path.mkdir(parents=True, exist_ok= False)
            # except FileExistsError as FEE:
            #     print(f'Path exists for {save_dir_path}')
            #     pass
            output_video_path = os.path.join(vid_out_path, 'Camera{}_out/'.format(ncam+1)+new_vid_name+'_{}.mp4'.format(vid_id))
        else:
            vid_out_path = viddir_path
            output_video_path = os.path.join(viddir_path, 'Camera{}_out/'.format(ncam+1)+new_vid_name+'_{}.mp4'.format(vid_id))

        # if not os.path.exists(os.path.join(viddir_path, 'Camera{}_out/'.format(ncam+1))):
        #     os.makedirs(os.path.join(viddir_path, 'Camera{}_out/'.format(ncam+1)))
        if not os.path.exists(os.path.join(vid_out_path, 'Camera{}_out/'.format(ncam+1))):
            os.makedirs(os.path.join(vid_out_path, 'Camera{}_out/'.format(ncam+1)))

        # Create a reader and writer
        reader = imageio.get_reader(input_video_path)
        # writer = imageio.get_writer(output_video_path, fps=reader.get_meta_data()['fps'])
        if fps is None:
            fps=reader.get_meta_data()['fps']
        # print("Current FPS = {}".format(reader.get_meta_data()['fps']))
        print("Writing FPS = {}, Video FPS = {}".format(fps, reader.get_meta_data()['fps']))

        metadata = dict(title='dannce_visualization', artist='Matplotlib')
        writer = FFMpegWriter(fps=fps, metadata=metadata)

        fig, axes = plt.subplots(1, 1, figsize=(8, 8), dpi=300)
        print("="*40 + "Starting for camera {}".format(ncam+1)+ "="*40)

        print("Saving video to ", output_video_path)
        with writer.saving(fig, output_video_path, dpi=300):

        # Iterate through frames, crop, and write to the new video
            # import pdb; pdb.set_trace()
            for frame_number in tqdm(range(start_sample, start_sample + max_samples)):
                # print("Frame Number = ",frame_number) 
                if len(sync[0]['data_frame'].shape) > 1 and len(sync[0]["data_sampleID"].shape) > 1 :
                    fr = sync[0]["data_frame"][(np.where(sync[0]["data_sampleID"] == predictions["sampleID"][0][frame_number]))]
                    # print('frame loc = ', (np.where(sync[0]["data_sampleID"] == predictions["sampleID"][0][frame_number])))
                else:
                    fr = sync[0]["data_frame"][(np.where(sync[0]["data_sampleID"] == predictions["sampleID"][0][frame_number]))[0].squeeze()]
                    # print('frame loc = ', (np.where(sync[0]["data_sampleID"] == predictions["sampleID"][0][frame_number]))[0])
                frame = reader.get_data(fr[0] - vid_id)
                
                # if cropped_frame.shape[0] != cropped_frame.shape[1]:
                #     print('x1 {}, y1 {}, x2 {}, y2 {}'.format(x1, y1, x2, y2))
                #     print("Image size = {}, {}".format(y2-y1, x2-x1))
                #     print("Cropped Frame Size = {}".format(cropped_frame.shape))
                
                axes.imshow(frame)

                imagePoints = predictions_2d[ncam][frame_number]
                
                if com_list != None:
                    com = com_list[ncam][frame_number]
                    axes.scatter(com[:,0], com[:,1], marker='.', color='red', linewidths=1)
                
                for mm in range(len(links)):
                    if links[mm][0] in goodmarks and links[mm][1] in goodmarks:
                        xx = [imagePoints[links[mm][0]-1,0],
                            imagePoints[links[mm][1]-1,0]]
                        yy = [imagePoints[links[mm][0]-1,1],
                            imagePoints[links[mm][1]-1,1]]

                        if annot and links[mm][0]-1 <= 16:
                            annotation = axes.annotate(str(joint_names[links[mm][0]-1][0]), (xx[0],yy[0]), 
                                    # bbox=dict(boxstyle="round4", fc="w"),
                                    )
                            annotation.set_alpha(0.4)
                            annotation.set_c('white')

                        axes.scatter(xx, yy, 
                                     marker = '.', 
                                     color=[bright_colors_list[links[mm][0]], 
                                            bright_colors_list[links[mm][1]]], 
                                     linewidths=0.3)
                        axes.plot(xx,yy, c=color_dict[mm], lw=1)
                
                axes.axis("off")
                axes.set_title(str(frame_number))
                
                # writer.append_data(cropped_frame)
                # axes[0].clear()
                # axes[1].clear()
                writer.grab_frame()
                axes.clear()

           



def plot_projected_points_part(predictions, 
                                sync, 
                                params, 
                                imagePoints_agg,
                                com_2d_agg, 
                                goodmarks, 
                                links, 
                                color_dict,
                                videofle_path,
                                video_save_path,
                                start_sample = 0, 
                                max_samples = 1000, 
                                fps = 30,
                            ):
  """
    # Plots the projected points and saves them to the locations specified in video_save_path
    # Both videofle_path and video_save_path except full paths with filename and extension.
    # This method is called from driver with all the related arguments passed.

    predictions: dict of predictions
    sync: dict required to sync frames from the video with the preductions. 
          This is necessary to determine which predictions correspons to which sample
    params: dict of params loaded from .mat file
    imagePoints_agg: list of lists of projection points for each camera view
    com_2d_agg: list of lists of projected Center of Mass for each camera view
    goodmarks: List of joint indices to consider while plotting
    links: List specifying which joint indices are connected to which with a bone
    color_dict: List of tupples mentioning colors for each bone
    videofle_path: Path from where to read video file
    video_save_path: Path to save videos to
    start_sample: SampleID to start reading frames from.
                  Default: 0
    max_samples: Max number of frames to read from video
                  Default: 1000
  """
  movie_reader = imageio.get_reader(videofle_path)

  metadata = dict(title='dannce_visualization', artist='Matplotlib')
  writer = FFMpegWriter(fps=fps, metadata=metadata)

  fig, axes = plt.subplots(1, 1, figsize=(8, 8), dpi=300)

  if not os.path.exists(os.path.dirname(video_save_path)):
    os.makedirs(os.path.dirname(video_save_path))

  with writer.saving(fig, video_save_path, dpi=300):

    for i in range(start_sample, start_sample + max_samples):

      # frame should be taken from sync[0]["data_frame"] from an index where data_sampleID from sync[0] matches sampleID at i-th index from predictions
      # using np.where for this gives a nested numpy array containing a single element(the index), so use squeeze     
      fr = sync[0]["data_frame"][(np.where(sync[0]["data_sampleID"] == predictions["sampleID"][0][i]))[0].squeeze()]
      frame = movie_reader.get_data(fr[0])
      print("Sample: ", i)
    
      axes.imshow(frame)      
      
      for ncam in range (len(params)):

        imagePoints = imagePoints_agg[ncam][i]
        if com_2d_agg != None:
          com = com_2d_agg[ncam][i]
          axes.scatter(com[:,0], com[:,1], marker='.', color='red', linewidths=1)
        
        for mm in range(len(links)):
          if links[mm][0] in goodmarks and links[mm][1] in goodmarks:
            xx = [imagePoints[links[mm][0]-1,0],
                  imagePoints[links[mm][1]-1,0]]
            yy = [imagePoints[links[mm][0]-1,1],
                  imagePoints[links[mm][1]-1,1]]

            axes.scatter(xx, yy, marker = '.', color='white', linewidths=0.5)
            axes.plot(xx,yy, c=color_dict[mm], lw=2)
        
        axes.axis("off")
        axes.set_title(str(i))
        
      writer.grab_frame()
      axes.clear()

def plot_projected_points_scratch(predictions, 
                          sync, 
                          params, 
                          imagePoints_agg,
                          com_2d_agg, 
                          goodmarks, 
                          links, 
                          color_dict,
                          videofle_path,
                          video_save_path,
                          joint_names = None,
                          start_sample = 0, 
                          max_samples = 1000, 
                          fps = 30,):
    
    movie_reader = imageio.get_reader(videofle_path)

    metadata = dict(title='dannce_visualization', artist='Matplotlib')
    writer = FFMpegWriter(fps=fps, metadata=metadata)

    fig, axes = plt.subplots(1, 1, figsize=(8, 8), dpi=300)

    if not os.path.exists(os.path.dirname(video_save_path)):
        os.makedirs(os.path.dirname(video_save_path))

    # import pdb; pdb.set_trace()

    with writer.saving(fig, video_save_path, dpi=300):

        for i in range(start_sample, start_sample + max_samples):
            fr = sync[0]["data_frame"][(np.where(sync[0]["data_sampleID"] == predictions["sampleID"][0][i]))[0].squeeze()]
            frame = movie_reader.get_data(fr[0])
            print("Sample: ", i)
            
            axes.imshow(frame) 

            imagePoints = imagePoints_agg[i]
            if not (com_2d_agg is None):
                com = com_2d_agg[i]
                # print(com)
                axes.scatter(com[0], com[1], marker='.', color='red', linewidths=1)

            for mm in range(len(links)):
                if links[mm][0] in goodmarks and links[mm][1] in goodmarks:
                    xx = [imagePoints[links[mm][0]-1,0],
                        imagePoints[links[mm][1]-1,0]]
                    yy = [imagePoints[links[mm][0]-1,1],
                        imagePoints[links[mm][1]-1,1]]

                    # import pdb; pdb.set_trace()
                    annotation = axes.annotate(str(joint_names[links[mm][0]-1][0]), (xx[0],yy[0]), 
                                # bbox=dict(boxstyle="round4", fc="w"),
                                )
                    annotation.set_alpha(0.4)
                    annotation.set_c('white')
                    axes.scatter(xx, yy, marker = '.', color='white', linewidths=0.5)
                    axes.plot(xx,yy, c=color_dict[mm], lw=2)
                
            axes.axis("off")
            axes.set_title(str(i))
        
            writer.grab_frame()
            axes.clear()

def get_bounding_box_3d_gen(predictions, com_3d, links, lims, col_dict):
    preds = predictions['pred']

    preds = np.concatenate([preds,
                            np.expand_dims(com_3d + [ -lims, -lims, -lims], 2),
                            np.expand_dims(com_3d + [ -lims, -lims,  lims], 2),
                            np.expand_dims(com_3d + [ -lims,  lims, -lims], 2),
                            np.expand_dims(com_3d + [ -lims,  lims,  lims], 2),
                            np.expand_dims(com_3d + [  lims, -lims, -lims], 2),
                            np.expand_dims(com_3d + [  lims, -lims,  lims], 2),
                            np.expand_dims(com_3d + [  lims,  lims, -lims], 2),
                            np.expand_dims(com_3d + [  lims,  lims,  lims], 2),],
                            axis=2)
    
    values = np.array([
                        [1, 2],
                        [2, 3],
                        [1, 3],
                        [2, 4],
                        [3, 4],
                        [5, 6],
                        [5, 7],
                        [6, 8],
                        [7, 8],
                        [1, 5],
                        [2, 6],
                        [3, 7],
                        [4, 8]
                    ]) + preds.shape[2]
    
    links = np.append(links, values, axis=0)

    col_dict.extend([
                        (0.8261, 0.4130, 0.1739, 0.5000),  # Light brown
                        (0.6957, 0.5217, 0.9565, 0.5000),  # Light purple
                        (0.5217, 0.6957, 0.9565, 0.5000),  # Light blue
                        (0.2609, 0.9565, 0.6957, 0.5000),  # Light green
                        (0.9565, 0.6957, 0.5217, 0.5000),  # Light orange
                        (0.9565, 0.5217, 0.6957, 0.5000),  # Light pink
                        (0.9565, 0.4130, 0.6957, 0.5000),  # Frank-1
                        (0.9565, 0.8261, 0.6957, 0.5000),  # Frank-2
                        (0.9565, 0.6217, 0.5957, 0.5000),  # Frank-3

                    ])

    return preds, links, col_dict

def augment_col_dict(links, col_dict):
    '''For the code to work length of links should be more or same as the length of links'''
    
    unwanted_colors = ['grey', 'white', 'snow', 'black', 'silver', 'gray']
    filtered_colors = [color for color in mcolors.CSS4_COLORS.keys() if not any(unwanted in color for unwanted in unwanted_colors)]
    if len(links) > len(col_dict):
        diff = len(links) - len(col_dict) + 1
        col_dict.extend([mcolors.to_rgba(filtered_colors[i], alpha=0.5) for i in range(diff)])
    
    return col_dict


def get_bounding_box_3d(predictions, com_3d, links, lims):
    ## TODO - make it so that you do not overwrite any pre-existing keypoints,
    # but generate a new set of 8 keypoints and links and colors 
    # The function should return a modified color_dict as well
    
    preds = predictions['pred']

    for frame in range(preds.shape[0]):
        # for i in range(len([16,17,18,19,20,21,22])):
        preds[frame,:,15] = com_3d[frame] + [-lims, -lims, -lims]
        preds[frame,:,16] = com_3d[frame] + [-lims, -lims,  lims]
        preds[frame,:,17] = com_3d[frame] + [-lims,  lims, -lims]
        preds[frame,:,18] = com_3d[frame] + [-lims,  lims,  lims]
        preds[frame,:,19] = com_3d[frame] + [ lims, -lims, -lims]
        preds[frame,:,20] = com_3d[frame] + [ lims, -lims,  lims]
        preds[frame,:,21] = com_3d[frame] + [ lims,  lims, -lims]
        
    preds = np.concatenate([preds,
                            np.expand_dims(com_3d + [ lims,  lims, lims], 2)],
                            axis=2)
    
    for idx,link in reversed(list(enumerate(links))):
        if np.isin(link,[16,17,18,19,20,21,22]).any():
            links=np.delete(links, idx, 0)
    
    values = np.array([
                        [16,17],
                        [17,18],
                        [16,18],
                        [17,19],
                        [18,19],
                        [20,21],
                        [20,22],
                        [21,23],
                        [22,23],
                        [16,20],
                        [17,21],
                        [18,22],
                        [19,23]
                    ])
    links = np.append(links, values, axis=0)
    
    return preds, links

def driver(viddir_path = '/hpc/group/tdunn/Bryan_Rigs/SmallOpenField/camera_calib_td/20240716_recording_data/1691486_face_left_0947/videos',
           label3d_filepath = '/hpc/group/tdunn/Bryan_Rigs/SmallOpenField/camera_calib_td/20240716_recording_data/1691486_face_left_0947/label3d_dannce.mat',
           predictions_2d_path = None #'/home/anshuman/ASDev/open_field_facemap/kps_2d.npy',
            ):
    '''
    This was initially written to use 2D keypoints predicted by facemap and plot them on DANNCE videos
    This was supposed to take care of all the cropping and other operations needed to obtain that.
    But currently (as of 12-04-2024), this is being used as a script to produce reprojections for 
    6 camera videos (can potentially also work with St3dio style videos).
    '''

    # preds_2D = np.load(predictions_2d_path)
    # head_KPs = preds_2D[:,:,:4,:]
    # head_COMs = np.mean(head_KPs, axis=2)

    # square_sides = get_max_sides_per_cam(head_KPs)

    dannceMat_filepath = '/hpc/group/tdunn/Bryan_Rigs/SmallOpenField/camera_calib_td/20240716_recording_data/1691486_face_left_0947/label3d_dannce.mat'
    # preditcions_filepath = '/home/anshuman/ASDev/dannce_release_dev2/demo/markerless_mouse_1/train_videos/DANNCE/predict_results/AS_SCR/MAY16_24/FACE_DANNCE_MNL1_AVGMAX_15mm_w_mse/save_data_AVG0.mat'
    # preditcions_filepath = '/home/anshuman/ASDev/dannce_release_dev2/demo/markerless_mouse_1/train_videos/DANNCE/predict_results/AS_SCR/MAY13_24/FACE_DANNCE_MNL1_MAX_15mm_w_mse/save_data_AVG0.mat'
    # preditcions_filepath = '/home/anshuman/ASDev/dannce_release_dev2/demo/markerless_mouse_1/train_videos/DANNCE/predict_results/AS_SCR/MAY10_24/FACE_DANNCE_GCE_MAX_15mm_w_mse/save_data_MAX0.mat'
    # preditcions_filepath = '/home/anshuman/ASDev/dannce_release_dev2/demo/markerless_mouse_1/train_videos/DANNCE/predict_results/AS_SCR/MAY11_24/FACE_DANNCE_MNL1_MAX_12mm_w_mse/save_data_MAX0.mat'
    # preditcions_filepath = '/home/anshuman/ASDev/dannce_release_dev2/demo/markerless_mouse_1/train_videos/DANNCE/predict_results/AS_SCR/JUN19_24/FACE_DANNCE_MNL1_AVGMAX_15mm_w_mse_FT_frm_AVG/save_data_AVG0.mat'
    # preditcions_filepath = '/home/anshuman/ASDev/dannce_release_dev2/demo/markerless_mouse_1/train_videos/DANNCE/predict_results/AS_SCR/JUN120_24/FACE_DANNCE_MNL1_AVGMAX100_24mm_w_mse_FT_frm_AVG/save_data_AVG0.mat'
    # preditcions_filepath = '/home/anshuman/ASDev/dannce_release_dev2/demo/markerless_mouse_1/DANNCE/predict_results/AS_SCR/JUN120_24/FACE_DANNCE_MNL1_AVGMAX100_24mm_w_mse_FT_frm_AVG/save_data_AVG0.mat'
    preditcions_filepath = '/hpc/group/tdunn/Bryan_Rigs/SmallOpenField/camera_calib_td/20240716_recording_data/1691486_face_left_0938/DANNCE/predict_results/AS_SCR/AUG08_24/FACE_DANNCE_MNL1_AVGMAX100_24mm_w_mse_FT_frm_AVG/save_data_MAX0.mat'
    skeleton_path = '/hpc/group/tdunn/asabath/dannce_scr/configs/left_or_right_colormap.mat'
    # skeleton_path = '/home/anshuman/ASDev/dannce_release_dev2/configs/face_kps.mat'
    # com3d_filepath = '/home/anshuman/ASDev/dannce_release_dev2/demo/markerless_mouse_1/train_videos/DANNCE/predict_results/AS_SCR/MAY16_24/FACE_DANNCE_MNL1_AVGMAX_15mm_w_mse/com3d_used.mat'
    # com3d_filepath = '/home/anshuman/ASDev/dannce_release_dev2/demo/markerless_mouse_1/COM/predict_results/AS_SCR/APR04_22/LCL_DEMO5V_01/face_com3d.mat'
    # com3d_filepath = '/hpc/group/tdunn/Bryan_Rigs/SmallOpenField/camera_calib_td/20240716_recording_data/1691486_face_left_0938/COM/predict_results/com3d.mat'
    com3d_filepath = '/hpc/group/tdunn/Bryan_Rigs/SmallOpenField/camera_calib_td/20240716_recording_data/1691486_face_left_0938/DANNCE/predict_results/AS_SCR/AUG08_24/FACE_DANNCE_MNL1_AVGMAX100_24mm_w_mse_FT_frm_AVG/com3d_used.mat'

    # exclude_joints = list(range(5,23))
    exclude_joints = [16,17,18,19,20,21,22]
    # exclude_joints = []

    cam_names, sync, params, skeleton, predictions, com_3d = get_data(dannceMat_filepath=label3d_filepath, 
                                                                    preditcions_filepath=preditcions_filepath, 
                                                                    skeleton_path=skeleton_path,
                                                                    com3d_filepath=com3d_filepath)
    
    #Trying to plot frames from a label3D_dannce file
    # l3d = dio.load_labels('/home/anshuman/ASDev/dannce_release_dev2/labellings/Finals/vid1/new_20240503_144420_Label3D_dannce.mat')
    # predictions['pred'] = np.transpose(l3d[1]['data_3d'].reshape(-1,22,3), (0,2,1))
    # predictions['data'] = np.zeros(predictions['pred'].shape)
    # predictions['sampleID'] = l3d[1]['data_sampleID'].T
    # com_3d = np.array([ # List comprehension
    #                     com_3d[ # Get the corresponding value from com_3d based on frame number
    #                         sync[0]['data_frame'][ # Get frame numbers where sampleID in sync match with sampleID in label3d
    #                                                 np.where(sync[0]['data_sampleID'] == l3d[1]['data_sampleID'][i])
    #                                             ][0].squeeze()
    #                             ] for i in range(len(l3d[1]['data_sampleID']))
    #                 ])
    col_dict = COLOR_DICT
    
    # _matfmt version of functions used for demo data processing - ToDo Unify them
    cameraParams, rot, trans, mirror, links, goodmarks = get_camParams_matfmt(params = params, skeleton = skeleton, 
                                                                            exclude_joints = exclude_joints)
    pred_3d = predictions['pred']
    # Set the last 7 keypoints as the bounding box for the head
    # predictions['pred'], links = get_bounding_box_3d(predictions, com_3d, links, lims=17)

    #Try Generalized bounding box script
    # predictions['pred'], links, col_dict = get_bounding_box_3d_gen(predictions, com_3d, links, lims=17, col_dict = COLOR_DICT)

    pose_3d = np.transpose(pred_3d, (0, 2, 1))
    # print("*"*10 + "Pose 3D = {}".format(pose_3d) + "*"*10)
    # head_coms_3d = np.mean(pose_3d[:,:3,:], axis=1)
    # _matfmt version of functions used for demo data processing - ToDo Unify them
    # imagePoints_agg, com_2d_agg = get_projected_points(predictions, params, cameraParams, rot, trans, None, com_3d)
    # imagePoints_agg, com_2d_agg = get_projected_points_matfmt(predictions, params, cameraParams, rot, trans, None, head_coms_3d)
    imagePoints_agg, com_2d_agg = get_projected_points_matfmt(
                                                                predictions, 
                                                                params, 
                                                                cameraParams, 
                                                                rot, 
                                                                trans, 
                                                                None, 
                                                                com_3d[:pose_3d.shape[0]]
                                                            )
   
    # ======================================================
    # Visualization code in this section works well for demo data
    # start_sample = 0
    # max_samples = 50
    # videofle_path = '/home/anshuman/ASDev/dannce_base/demo/markerless_mouse_1/videos/Camera1/0.mp4'
    # video_save_path = '/home/anshuman/ASDev/dannce_base/demo/markerless_mouse_1/videos/Camera1/0_headonly.mp4'
    # videofle_path = '/home/anshuman/ASDev/dannce_base/demo/markerless_mouse_1/videos/Camera2/0.mp4'
    # video_save_path = '/home/anshuman/ASDev/dannce_base/demo/markerless_mouse_1/videos/Camera2/0_headonly.mp4'
    # fps=30
    # plot_projected_points_scratch(predictions, 
    #                     sync, 
    #                     params, 
    #                     imagePoints_agg[0],
    #                     None, 
    #                     goodmarks, 
    #                     links, 
    #                     skeleton["color"],
    #                     videofle_path,
    #                     video_save_path,
    #                     joint_names = skeleton['joint_names'],
    #                     start_sample = start_sample,
    #                     max_samples = max_samples,
    #                     fps=fps,
    #                 )
    #=========================================================
    # preds_2D = np.array(imagePoints_agg)
    # head_KPs = preds_2D[:,:,:5,:]
    # import pdb; pdb.set_trace()
    # head_COMs = ((head_KPs[:,:,4,:] + head_KPs[:,:,0,:])/2)
    # head_COMs = np.mean(np.stack([head_KPs[:,:,3,:],head_KPs[:,:,0,:]], axis=2), axis=2)
    # head_COMs = np.mean(head_KPs[:,:,:3,:], axis=2)
    # head_COMs = np.array(com_2d_agg).squeeze()

    # square_sides = get_max_sides_per_cam(head_KPs)

    # ======================================================
    # Visualize Head COMs
    start_sample = 0
    # max_samples = 50
    max_samples = 100
    # videofle_path = '/home/anshuman/ASDev/dannce_base/demo/markerless_mouse_1/videos/Camera1/0.mp4'
    # video_save_path = '/home/anshuman/ASDev/dannce_base/demo/markerless_mouse_1/videos/Camera1/0_headonly.mp4'
    # videofle_path = '/home/anshuman/ASDev/dannce_base/demo/markerless_mouse_1/videos/Camera2/0.mp4'
    # video_save_path = '/home/anshuman/ASDev/dannce_base/demo/markerless_mouse_1/videos/Camera2/0_headonly.mp4'
    # fps=30
    # plot_projected_points_scratch(predictions, 
    #                     sync, 
    #                     params, 
    #                     imagePoints_agg[1],
    #                     head_COMs[1], 
    #                     goodmarks, 
    #                     links, 
    #                     skeleton["color"],
    #                     videofle_path,
    #                     video_save_path,
    #                     joint_names = skeleton['joint_names'],
    #                     start_sample = start_sample,
    #                     max_samples = max_samples,
    #                     fps=fps,
    #                 )
    #=========================================================

    # crop_vid(predictions_2d = preds_2D,
    #          viddir_path = viddir_path,
    #          square_side = square_sides,
    #          com_list = head_COMs,
    #          params = params,
    #          sync = sync,
    #          predictions= predictions,
    #          start_sample = 0,
    #          max_samples= 1000,
    #          new_vid_name="cropped"
    #          )

    
    print(len(links))
    print(len(col_dict))
    print("Shape of ImagePoints = ", imagePoints_agg[0].shape)

    full_bod_vid(
                    imagePoints_agg,
                    viddir_path,
                    com_2d_agg,
                    params,
                    sync,
                    predictions,
                    links,
                    goodmarks,
                    joint_names=skeleton['joint_names'], 
                    color_dict=col_dict,
                    start_sample = 0,
                    max_samples=100,
                    # new_vid_name="AUG08_24_TEST",
                    new_vid_name="MAY12_25_TEST",
                    # new_vid_name="JUN120_24_FACE_DANNCE_MNL1_AVGMAX100_24mm_w_mse_FT_frm_AVG_fullfps_thin",
                    # fps=1,
                    vid_id=0,
                    annot=True,
                )


def cleaned_driver(viddir_path = '/hpc/group/tdunn/Bryan_Rigs/SmallOpenField/camera_calib_td/20240716_recording_data/1691486_face_left_0947/videos',
                    label3d_filepath = '/hpc/group/tdunn/Bryan_Rigs/SmallOpenField/camera_calib_td/20240716_recording_data/1691486_face_left_0947/label3d_dannce.mat',
                    preditcions_filepath = '/hpc/group/tdunn/Bryan_Rigs/SmallOpenField/camera_calib_td/20240716_recording_data/1691486_face_left_0938/DANNCE/predict_results/AS_SCR/AUG08_24/FACE_DANNCE_MNL1_AVGMAX100_24mm_w_mse_FT_frm_AVG/save_data_MAX0.mat',
                    skeleton_path = '/hpc/group/tdunn/asabath/dannce_scr/configs/left_or_right_colormap.mat',
                    com3d_filepath = '/hpc/group/tdunn/Bryan_Rigs/SmallOpenField/camera_calib_td/20240716_recording_data/1691486_face_left_0938/DANNCE/predict_results/AS_SCR/AUG08_24/FACE_DANNCE_MNL1_AVGMAX100_24mm_w_mse_FT_frm_AVG/com3d_used.mat',
                    exclude_joints = [16,17,18,19,20,21,22],
                    start_sample = 0,
                    max_samples = 50,
                    vid_name_append="AUG08_24_TEST",
                    vid_out_path=None,
                    subtract_com=False
            ):
    cam_names, sync, params, skeleton, predictions, com_3d = get_data(dannceMat_filepath=label3d_filepath, 
                                                                    preditcions_filepath=preditcions_filepath, 
                                                                    skeleton_path=skeleton_path,
                                                                    com3d_filepath=com3d_filepath)
    
    
    col_dict = COLOR_DICT
    
    # _matfmt version of functions used for demo data processing - ToDo Unify them
    cameraParams, rot, trans, mirror, links, goodmarks = get_camParams_matfmt(params = params, skeleton = skeleton, 
                                                                            exclude_joints = exclude_joints)
    pred_3d = predictions['pred']

    pose_3d = np.transpose(pred_3d, (0, 2, 1))
    imagePoints_agg, com_2d_agg = get_projected_points_matfmt(
                                                                predictions, 
                                                                params, 
                                                                cameraParams, 
                                                                rot, 
                                                                trans, 
                                                                None, 
                                                                com_3d[:pose_3d.shape[0]],
                                                                subtract_com=subtract_com
                                                            )

    
    print(len(links))
    print(len(col_dict))
    print("Joint names len = {}".format(len(skeleton['joint_names'])))
    print("Shape of ImagePoints = ", imagePoints_agg[0].shape)
    print('Shape of com2d = ', com_2d_agg[0].shape)

    com_corrected_impts = [impts - com_2d_agg[i] for i,impts in enumerate(imagePoints_agg)]

    if len(links) != len(col_dict):
        print('='*50 + " Augmenting Color Dict " + "="*50)
        print("ColorDictAugmentWarning: To prevent this, make sure the color dict specified is the same length as links ")
        col_dict = augment_col_dict(links, col_dict)

    full_bod_vid(
                    imagePoints_agg,
                    # com_corrected_impts,
                    viddir_path,
                    com_2d_agg,
                    params,
                    sync,
                    predictions,
                    links,
                    goodmarks,
                    joint_names=skeleton['joint_names'], 
                    color_dict=col_dict,
                    start_sample = start_sample,
                    max_samples=max_samples,
                    new_vid_name=vid_name_append,
                    # new_vid_name="JUN120_24_FACE_DANNCE_MNL1_AVGMAX100_24mm_w_mse_FT_frm_AVG_fullfps_thin",
                    fps=15,
                    vid_id=0,
                    annot=False,
                    vid_out_path=vid_out_path,
                )

# driver()
# cleaned_driver(viddir_path = '/hpc/group/tdunn/Bryan_Rigs/SmallOpenField/camera_calib_td/20240625_recording_data/1691485_left/videos',
#                 label3d_filepath = '/hpc/group/tdunn/Bryan_Rigs/SmallOpenField/camera_calib_td/20240625_recording_data/1691485_left/label3d_dannce.mat',
#                 preditcions_filepath = '/hpc/group/tdunn/Bryan_Rigs/SmallOpenField/camera_calib_td/20240625_recording_data/1691485_left/DANNCE/predict_results/twd5/smoothed_prediction_twd5_medfilt5.mat',
#                 skeleton_path = '/hpc/group/tdunn/asabath/dannce_scr/configs/mouse22_skeleton.mat',
#                 com3d_filepath = '/hpc/group/tdunn/Bryan_Rigs/SmallOpenField/camera_calib_td/20240625_recording_data/1691485_left/DANNCE/predict_results/twd5/com3d_used.mat',
#                 exclude_joints = [],
#                 start_sample = 0,
#                 max_samples = 1000,
#                 vid_name_append = "2024DEC05"
#                 )

#### ACTUAL WORKING CODE ### (The above call also works potentially, but did not verify)#
## Commented so that the file can be imported
# cleaned_driver(viddir_path = '/hpc/group/tdunn/ami25/dannce_reldev2/demo/20191028_mouse6/videos',
#                 label3d_filepath = '/hpc/group/tdunn/ami25/dannce_reldev2/demo/20191028_mouse6/20210613_221425_Label3D_dannce.mat',
#                 preditcions_filepath = '/hpc/group/tdunn/ami25/dannce_reldev2/demo/20191028_mouse6/DANNCE/predict_results/save_data_AVG.mat',
#                 skeleton_path = '/hpc/group/tdunn/asabath/dannce_scr/configs/mouse22_skeleton.mat',
#                 com3d_filepath = '/hpc/group/tdunn/ami25/dannce_reldev2/demo/20191028_mouse6/DANNCE/predict_results/com3d_used.mat',
#                 exclude_joints = [],
#                 start_sample = 0,
#                 max_samples = 1000,
#                 vid_name_append = "2024DEC05"
#                 )

# This worked for facial keypoints prediction in that specific video
# cleaned_driver(viddir_path = '/hpc/group/tdunn/Bryan_Rigs/SmallOpenField/camera_calib_td/20240628_recording_data/1691486_both/videos',
#                 label3d_filepath = '/hpc/group/tdunn/Bryan_Rigs/SmallOpenField/camera_calib_td/20240628_recording_data/1691486_both/label3d_dannce.mat',
#                 preditcions_filepath = '/hpc/group/tdunn/Bryan_Rigs/SmallOpenField/camera_calib_td/20240628_recording_data/1691486_both/DANNCE/predict_results/AS_SCR/AUG09_24/AVG_FTSeg_avgmax100_24mm_mse_dist3D/save_data_AVG0.mat',
#                 skeleton_path = '/hpc/group/tdunn/Bryan_Rigs/SmallOpenField/dannce/configs/face_kps.mat',
#                 com3d_filepath = '/hpc/group/tdunn/Bryan_Rigs/SmallOpenField/camera_calib_td/20240628_recording_data/1691486_both/DANNCE/predict_results/AS_SCR/AUG09_24/AVG_FTSeg_avgmax100_24mm_mse_dist3D/com3d_used.mat',
#                 exclude_joints = [16,17,18,19,20,21,22],
#                 start_sample = 0,
#                 max_samples = 1000,
#                 vid_name_append = "2025MAY12",
#                 )

# cleaned_driver(viddir_path = '/hpc/group/tdunn/Bryan_Rigs/SmallOpenField/camera_calib_td/20240718_recording_data/1686940_timepoint2_left_1020/videos',
#                 label3d_filepath = '/hpc/group/tdunn/Bryan_Rigs/SmallOpenField/camera_calib_td/20240718_recording_data/1686940_timepoint2_left_1020/label3d_dannce.mat',
#                 preditcions_filepath = '/hpc/group/tdunn/Bryan_Rigs/SmallOpenField/camera_calib_td/20240718_recording_data/1686940_timepoint2_left_1020/DANNCE/predict_results/AS_SCR/AUG09_24_AVG_FTSeg_avgmax100_24mm_mse_dist3D_medfilted/save_data_AVG.mat',
#                 skeleton_path = '/hpc/group/tdunn/Bryan_Rigs/SmallOpenField/dannce/configs/face_kps.mat',
#                 com3d_filepath = '/hpc/group/tdunn/Bryan_Rigs/SmallOpenField/camera_calib_td/20240718_recording_data/1686940_timepoint2_left_1020/DANNCE/predict_results/AS_SCR/AUG09_24_AVG_FTSeg_avgmax100_24mm_mse_dist3D/com3d_used.mat',
#                 exclude_joints = [16,17,18,19,20,21,22],
#                 start_sample = 0,
#                 max_samples = 100,
#                 vid_name_append = "2025MAY12",
#                 )

# cleaned_driver(viddir_path = '/hpc/group/tdunn/Bryan_Rigs/SmallOpenField/camera_calib_td/20250403_recording_data/1686941_both_20test1/videos',
#                 label3d_filepath = '/hpc/group/tdunn/Bryan_Rigs/SmallOpenField/camera_calib_td/20250403_recording_data/1686941_both_20test1/label3d_dannce.mat',
#                 preditcions_filepath = '/hpc/group/tdunn/Bryan_Rigs/SmallOpenField/camera_calib_td/20250403_recording_data/1686941_both_20test1/DANNCE/predict_results/AS_SCR/AUG09_24_AVG_FTSeg_avgmax100_24mm_mse_dist3D_medfilted/smoothed_prediction_AVG_medfilt5.mat',
#                 skeleton_path = '/hpc/group/tdunn/Bryan_Rigs/SmallOpenField/dannce/configs/face_kps.mat',
#                 com3d_filepath = '/hpc/group/tdunn/Bryan_Rigs/SmallOpenField/camera_calib_td/20250403_recording_data/1686941_both_20test1/DANNCE/predict_results/AS_SCR/AUG09_24_AVG_FTSeg_avgmax100_24mm_mse_dist3D_medfilted/com3d_used.mat',
#                 exclude_joints = [17,18,19,20,21,22],
#                 start_sample = 0,
#                 max_samples = 100,
#                 vid_name_append = "2025MAY13",
#                 )

