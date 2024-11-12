import torch
import time
import os

import numpy as np

import pdb

import matplotlib.pyplot as plt
from matplotlib.patches import Circle

from prediction.model.predictor import FullSegformerCustomHead
from prediction.config import get_config
from prediction.data.prepare_loader import prepare_dataloaders

from einops import rearrange

from tqdm import tqdm

if __name__ == '__main__':
    
        
    cfg, hparams = get_config(cfg_dir='./prediction/configs/', 
                     cfg_file='tiny_short')
    
    pdb.set_trace()


    
    device = torch.device('cpu')
    
    cfg.DATASET.VERSION = "v1.0-mini"
    cfg.N_WORKERS = 0
    cfg.BATCHSIZE = 1

    _ , valloader, _, val_data = prepare_dataloaders(cfg, return_dataset=True,
                                        return_orig_images=True, return_pcl=True)
    
    receptive_field = 1
    DOWNSAMPLE = 8
    MAX_DEPTH = 50
        
    ID = 20
    print('ID:',ID, 'Val data:',len(val_data))
    data = val_data[ID]
    full_image = data['image'].unsqueeze(0)[:, :receptive_field].contiguous().to(device)
    intrinsics = data['intrinsics'].unsqueeze(0)[:, :receptive_field].contiguous().to(device)
    extrinsics = data['extrinsics'].unsqueeze(0)[:, :receptive_field].contiguous().to(device)
    future_egomotion = data['future_egomotion'].unsqueeze(0)[:, :receptive_field].contiguous().to(device)
    lidar_ego_2_cam = data['lidar_ego_2_cam'].unsqueeze(0)[:, :receptive_field].contiguous().to(device)
    lidar_sensor_2_ego = data['lidar_sensor_2_ego'].unsqueeze(0)[:, :receptive_field].contiguous().to(device)
    # lidar_pcl = data['lidar_pcl'].unsqueeze(0)[:, :receptive_field].contiguous().to(device)


    out = model(full_image,intrinsics,extrinsics,future_egomotion)
        
    exit()
   
   
    # Filter padded points in LiDAR
    lidar_pcl = lidar_pcl[:, :receptive_field].contiguous().to(device)
    pt_filter = torch.logical_and(lidar_pcl[..., 0] != 0,
                                    lidar_pcl[..., 1] != 0).squeeze()
    lidar_pcl = lidar_pcl[...,pt_filter,:].squeeze()

    # Calculate depth parameters
    depth_bounds = cfg.LIFT.D_BOUND
    depth_categories = torch.arange(*depth_bounds).to(device)
          

    for cam in range(6):    
        # Project into 3d cam reference and filter behind the corresponding camera
        local_pcl = lidar_pcl.clone()
        local_pcl = torch.cat((local_pcl,torch.ones_like(local_pcl[:,0]).unsqueeze(1)),1).T
        local_pcl = lidar_ego_2_cam[0,0,cam] @ lidar_sensor_2_ego[0,0] @ local_pcl

        local_pcl = local_pcl[:,local_pcl[2,:] > 0]
        
        # Project LiDAR in cam 3d coord system into image
        exp_intrinsics = torch.eye(4)
        exp_intrinsics[:3,:3] = intrinsics[0,0,cam]
        
        projection = exp_intrinsics @ local_pcl
        projection[:3,:] = projection[:3,:] / projection[2]
        
        # Add 3D info
        projection = torch.cat((projection,local_pcl),0)
        
        # Filter points outside the image 
        IMG_HEIGHT, IMG_WIDTH = full_image.shape[-2:]
            
        pt_filter = torch.logical_and(projection[0,:] > 0,
                                    projection[1,:] > 0)
        pt_filter = torch.logical_and(pt_filter,
                                    projection[1,:] < IMG_HEIGHT)
        pt_filter = torch.logical_and(pt_filter,
                                    projection[0,:] < IMG_WIDTH).squeeze()
        projection = projection[:,pt_filter]
        
        # Create tensor for LiDAR points
        depth_map = torch.ones((IMG_HEIGHT,IMG_WIDTH)) * (MAX_DEPTH + 1)
        depth_values = torch.norm(projection[4:7,:],dim=0)
        
        depth_map[projection[1,:].long(),projection[0,:].long()] = depth_values
        depth_map = depth_map.unsqueeze(0)
        
        depth_map = -torch.nn.functional.max_pool2d(-depth_map,
                                                    kernel_size=DOWNSAMPLE,
                                                    stride=DOWNSAMPLE)
        pdb.set_trace()

        # Assign categories to each pixel (TODO: check if -1 is neccesary to match with the model categories)
        depth_map[depth_map[:]>MAX_DEPTH] = -1 
        depth_cat_map = torch.zeros_like(depth_map,dtype=torch.long)
        depth_cat_map[depth_map[:]>0] = torch.bucketize(depth_map[depth_map[:]>0],
                                                        depth_categories) - 1
        
        # Save depth_map as heatmap
        # plt.imshow(depth_map.squeeze(), cmap='hot', interpolation='nearest')
        # plt.tight_layout(pad=0)
        # plt.axis('off')
        # plt.margins(0)
        # plt.gca().xaxis.set_major_locator(plt.NullLocator())
        # plt.gca().yaxis.set_major_locator(plt.NullLocator())
        # plt.savefig(f'depth_map_heatmap_cam{cam}.png', bbox_inches='tight', pad_inches=0)
        # plt.close()
        
        # Plot and save the image
        image_np = full_image[0, 0, cam].permute(1, 2, 0).cpu().numpy()
        
        fig, ax = plt.subplots(figsize=(16, 9))
        ax.imshow(image_np)
        ax.axis('off')  # Turn off the axis
        plt.tight_layout()
        plt.axis('off')
        
        # for i in tqdm(range(projection.shape[1])):
        #     circ = Circle((projection[0, i], projection[1, i]), 1,
        #                   color=plt.cm.plasma((projection[-2, i]/MAX_DEPTH).cpu().numpy()))
        #     ax.add_patch(circ)
                    
        plt.savefig(f'projection_image{cam}.png', bbox_inches='tight', pad_inches=0)
        
        plt.close(fig)



    # del model
    # torch.cuda.reset_peak_memory_stats()
    # torch.cuda.empty_cache()
