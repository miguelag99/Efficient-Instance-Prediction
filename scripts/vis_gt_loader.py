import cv2
import os
import numpy as np
import torch
import matplotlib as mpl
import matplotlib.pyplot as plt
import torchvision

from PIL import Image

from nuscenes.nuscenes import NuScenes

from tqdm import tqdm

import sys
sys.path.append('../')

from prediction.data.nuscenes_dataset import NuscenesDataset
from prediction.configs import baseline_cfg
from prediction.utils.instance import generate_gt_instance_segmentation
                                       
from prediction.utils.network import NormalizeInverse
from prediction.utils.geometry import calculate_birds_eye_view_parameters
from prediction.utils.visualisation import (convert_figure_numpy,
                                            generate_instance_colours,
                                            make_contour, plot_instance_map)


if __name__ == '__main__':
    
    cfg = baseline_cfg
    cfg.DATASET.DATAROOT = '/home/perception/Datasets/nuscenes/trainval'
    cfg.DATASET.VERSION = 'v1.0-mini'
    
    device = torch.device('cuda:0')
    mode = 'val'
    
    save_path = os.path.join('plots', f'{mode}')
    os.mkdir(save_path, exist_ok=True)
   
    nusc = NuScenes(
        version=cfg.DATASET.VERSION,
        dataroot=cfg.DATASET.DATAROOT,
        verbose=True
    )
    
    ego_dims = (4.087,1.562,1.787)
    
    dl = NuscenesDataset(cfg, mode, return_orig_images=True)
            
    # Bird's-eye view parameters
    bev_resolution, bev_start_position, bev_dimension = calculate_birds_eye_view_parameters(
        cfg.LIFT.X_BOUND, cfg.LIFT.Y_BOUND, cfg.LIFT.Z_BOUND
    )
    bev_resolution, bev_start_position, bev_dimension = (
        bev_resolution.numpy(), bev_start_position.numpy(), bev_dimension.numpy()
    )
    
    for i,batch in enumerate(tqdm(dl)):
        
        segmentation = batch['segmentation']
        flow = batch['flow']
        centerness = batch['centerness']
        image = batch['orig_image']

        # Add batch dimension
        image = image.unsqueeze(0)
        segmentation = segmentation.unsqueeze(0)
        flow = flow.unsqueeze(0)
        centerness = centerness.unsqueeze(0)        
        
        time_range = cfg.TIME_RECEPTIVE_FIELD
        data = {
            'segmentation': segmentation[:, time_range:],
            'instance_flow': flow[:, time_range:],
            'centerness': centerness[:, time_range:],
        }
        
        # Process ground truth
        consistent_instance_seg, matched_centers = generate_gt_instance_segmentation(
            data, compute_matched_centers=True,
            spatial_extent=(cfg.LIFT.X_BOUND[1], cfg.LIFT.Y_BOUND[1])
        )
        
        first_instance_seg = consistent_instance_seg[0, 1]

        # Plot future trajectories
        unique_ids = torch.unique(first_instance_seg).cpu().long().numpy()[1:]
        instance_map = dict(zip(unique_ids, unique_ids))
        instance_colours = generate_instance_colours(instance_map)
        vis_image = plot_instance_map(first_instance_seg.cpu().numpy(), instance_map)
        trajectory_img = np.zeros(vis_image.shape, dtype=np.uint8)
        for instance_id in unique_ids:
            path = matched_centers[instance_id]
            for t in range(len(path) - 1):
                color = instance_colours[instance_id].tolist()
                cv2.line(trajectory_img, tuple(map(int,path[t])), tuple(map(int,path[t+1])),
                        color, 4)

        # Overlay arrows
        temp_img = cv2.addWeighted(vis_image, 0.7, trajectory_img, 0.3, 0.0)
        mask = ~ np.all(trajectory_img == 0, axis=2)
        vis_image[mask] = temp_img[mask]
        
        # Plot ego pose at the center of the image with cv2 circle       
        pts = np.array([[ego_dims[1]/2, ego_dims[0]/2],
                        [ego_dims[1]/2, -ego_dims[0]/2],
                        [-ego_dims[1]/2, -ego_dims[0]/2],
                        [-ego_dims[1]/2, ego_dims[0]/2]])
        
        pts = np.round((pts - bev_start_position[:2] + bev_resolution[:2] / 2.0) / bev_resolution[:2]).astype(np.int32)
        vis_image = cv2.fillPoly(vis_image, [pts], (0, 0, 0))
    
        # Plot present RGB frames and predictions
        val_w = 4.99
        cameras = cfg.IMAGE.NAMES
        image_ratio = cfg.IMAGE.ORIGINAL_DIM[0] / cfg.IMAGE.ORIGINAL_DIM[1]
        val_h = val_w * image_ratio
        fig = plt.figure(figsize=(4 * val_w, 2 * val_h))
        width_ratios = (val_w, val_w, val_w, val_w)
        gs = mpl.gridspec.GridSpec(2, 4, width_ratios=width_ratios)
        gs.update(wspace=0.0, hspace=0.0, left=0.0, right=1.0, top=1.0, bottom=0.0)

        denormalise_img = torchvision.transforms.Compose(
            (# NormalizeInverse(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            torchvision.transforms.ToPILImage(),)
        )
        # for imgi, img in enumerate(image[0, -1]):
        for imgi, img in enumerate(image[0, time_range-1]):
            ax = plt.subplot(gs[imgi // 3, imgi % 3])
            showimg = denormalise_img(img.cpu())
            if imgi > 2:
                showimg = showimg.transpose(Image.FLIP_LEFT_RIGHT)

            plt.annotate(cameras[imgi].replace('_', ' ').replace('CAM ', ''), (0.01, 0.87), c='white',
                        xycoords='axes fraction', fontsize=14)
            plt.imshow(showimg)
            plt.axis('off')


        ax = plt.subplot(gs[:, 3])
        
        plt.imshow(make_contour(vis_image[::-1, ::-1]))
        plt.axis('off')

        plt.draw()
        out_frame = convert_figure_numpy(fig)
        plt.close()
        
        out_frame = Image.fromarray(out_frame)
        out_frame.save(os.path.join(save_path,f"scene_{i}.png"))

        
