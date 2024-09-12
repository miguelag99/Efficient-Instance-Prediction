import argparse
import os

import cv2
import numpy as np
import torch
from tqdm import tqdm

from prediction.data.prepare_loader import prepare_dataloaders
from prediction.trainer import TrainingModule
from prediction.utils.geometry import calculate_birds_eye_view_parameters
from prediction.utils.instance import (
    generate_gt_instance_segmentation,
    predict_instance_segmentation,
)
from prediction.utils.visualisation import (
    generate_instance_colours,
    make_contour,
    plot_instance_map,
)

EGO_DIMS = (4.087,1.562,1.787)

def main(args):

    ckpt = args.checkpoint
    assert os.path.exists(ckpt),\
        f'Checkpoint {ckpt} does not exist'
        
    l_module = TrainingModule.load_from_checkpoint(ckpt)
    
    cfg = l_module.cfg
    cfg.DATASET.DATAROOT = args.dataset_root
    cfg.DATASET.VERSION = args.nusc_version
    
    print(f'Loading validation dataset from {cfg.DATASET.DATAROOT}' + \
        'version {cfg.DATASET.VERSION}')
    _, valloader, _, valdataset = prepare_dataloaders(cfg, return_dataset=True,
                                                      return_orig_images=True)
    

    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')
    
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    
    
    # Load model weights from checkpoint
    l_module = TrainingModule.load_from_checkpoint(ckpt)
    print(f'Loaded model from {ckpt}')
    l_module = l_module.to(device)
    l_module.eval()
        
    # Plot only one sequence
    if args.seq_id != -1:
        batch = valdataset[args.seq_id]
        for key in batch:
            batch[key] = batch[key].unsqueeze(0)
        
        output = l_module.model(batch['image'].to(device),
                                batch['intrinsics'].to(device),
                                batch['extrinsics'].to(device),
                                batch['future_egomotion'].to(device))
        
        generate_val_plots(cfg, batch, output, args.seq_id,
                           save_path = args.save_path, save_images = args.save_images)
        
            
        exit()
    
    # Iterate over all val dataset
    for i, batch in tqdm(enumerate(valdataset), total=len(valdataset)):
        
        # Add batch dimension
        for key in batch:
            batch[key] = batch[key].unsqueeze(0)
        
        output = l_module.model(batch['image'].to(device),
                                batch['intrinsics'].to(device),
                                batch['extrinsics'].to(device),
                                batch['future_egomotion'].to(device))
        
        generate_val_plots(cfg, batch, output, i, save_path = args.save_path,
                           save_images = args.save_images)
        
        
                

def generate_val_plots(cfg, batch, output, seq_id,
                       save_path = 'results', save_images = False):
    
        # Bird's-eye view parameters
        bev_resolution, bev_start_position, bev_dimension = calculate_birds_eye_view_parameters(
            cfg.LIFT.X_BOUND, cfg.LIFT.Y_BOUND, cfg.LIFT.Z_BOUND
        )
        bev_resolution, bev_start_position, bev_dimension = (
            bev_resolution.numpy(), bev_start_position.numpy(), bev_dimension.numpy()
        )
    
        save_path = os.path.join(save_path,'val', f'{seq_id}')
        os.makedirs(save_path, exist_ok=True)
    
        time_range = cfg.TIME_RECEPTIVE_FIELD
    
        if save_images:
            # Save input images
            for i,im in enumerate(batch['orig_image'][0,time_range-1]):
                im = im.permute(1, 2, 0).numpy()
                im = (im * 255).astype(np.uint8)
                im = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
                cv2.imwrite(os.path.join(save_path,f'image_{i}.png'), im)
            
        # Generate gt instance prediction
        data = {
            'segmentation': batch['segmentation'][:, time_range:],
            'instance_flow': batch['flow'][:, time_range:],
            'centerness': batch['centerness'][:, time_range:],
        }
        consistent_instance_seg, matched_centers = generate_gt_instance_segmentation(
            data, compute_matched_centers=True,
            spatial_extent=(cfg.LIFT.X_BOUND[1], cfg.LIFT.Y_BOUND[1])
        )
        
        first_instance_seg = consistent_instance_seg[0, 1]
        
        unique_ids = torch.unique(first_instance_seg).cpu().long().numpy()[1:]
        instance_map = dict(zip(unique_ids, unique_ids))
        instance_colours = generate_instance_colours(instance_map)
        vis_image = plot_instance_map(first_instance_seg.cpu().numpy(), instance_map)
        trajectory_img = np.zeros(vis_image.shape, dtype=np.uint8)
        for instance_id in unique_ids:
            path = matched_centers[instance_id]
            for t in range(len(path) - 1):
                color = instance_colours[instance_id].tolist()
                cv2.line(trajectory_img, tuple(map(int,path[t])),
                         tuple(map(int,path[t+1])), color, 4)

        # Overlay arrows
        temp_img = cv2.addWeighted(vis_image, 0.7, trajectory_img, 0.3, 0.0)
        mask = ~ np.all(trajectory_img == 0, axis=2)
        vis_image[mask] = temp_img[mask]
        
        # Plot ego pose at the center of the image with cv2 circle       
        pts = np.array([[EGO_DIMS[1]/2, EGO_DIMS[0]/2],
                        [EGO_DIMS[1]/2, -EGO_DIMS[0]/2],
                        [-EGO_DIMS[1]/2, -EGO_DIMS[0]/2],
                        [-EGO_DIMS[1]/2, EGO_DIMS[0]/2]])
        
        pts = np.round((pts - bev_start_position[:2] + bev_resolution[:2] / 2.0) / bev_resolution[:2]).astype(np.int32)
        vis_image = cv2.fillPoly(vis_image, [pts], (0, 0, 0))
        vis_image =cv2.cvtColor(vis_image, cv2.COLOR_RGB2BGR)
        
        cv2.imwrite(os.path.join(save_path,'gt_instance_seg.png'),
                    make_contour(vis_image[::-1, ::-1]))
        
        # Generate predicted instance prediction
                
        output['segmentation'] = output['segmentation'][0,1:,:,:].unsqueeze(0)
        output['instance_flow'] = output['instance_flow'][0,1:,:,:].unsqueeze(0)
        
        consistent_instance_seg, matched_centers = predict_instance_segmentation(
            output, compute_matched_centers=True,
            spatial_extent=(cfg.LIFT.X_BOUND[1],cfg.LIFT.Y_BOUND[1])
        )
        
        first_instance_seg = consistent_instance_seg[0, 1]
        
        unique_ids = torch.unique(first_instance_seg).cpu().long().numpy()[1:]
        instance_map = dict(zip(unique_ids, unique_ids))
        instance_colours = generate_instance_colours(instance_map)
        vis_image = plot_instance_map(first_instance_seg.cpu().numpy(), instance_map)
        trajectory_img = np.zeros(vis_image.shape, dtype=np.uint8)
        for instance_id in unique_ids:
            path = matched_centers[instance_id]
            for t in range(len(path) - 1):
                color = instance_colours[instance_id].tolist()
                cv2.line(trajectory_img, tuple(map(int,path[t])),
                         tuple(map(int,path[t+1])), color, 4)

        # Overlay arrows
        temp_img = cv2.addWeighted(vis_image, 0.7, trajectory_img, 0.3, 0.0)
        mask = ~ np.all(trajectory_img == 0, axis=2)
        vis_image[mask] = temp_img[mask]
        
        # Plot ego pose at the center of the image with cv2 circle       
        pts = np.array([[EGO_DIMS[1]/2, EGO_DIMS[0]/2],
                        [EGO_DIMS[1]/2, -EGO_DIMS[0]/2],
                        [-EGO_DIMS[1]/2, -EGO_DIMS[0]/2],
                        [-EGO_DIMS[1]/2, EGO_DIMS[0]/2]])
        
        pts = np.round((pts - bev_start_position[:2] + bev_resolution[:2] / 2.0) / bev_resolution[:2]).astype(np.int32)
        vis_image = cv2.fillPoly(vis_image, [pts], (0, 0, 0))
        vis_image =cv2.cvtColor(vis_image, cv2.COLOR_RGB2BGR)
        
        cv2.imwrite(os.path.join(save_path,'pred_instance_seg.png'),
                    make_contour(vis_image[::-1, ::-1]))
             
    
if __name__ == "__main__":
    # Create parser with one argument
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str,required=True,
                        default='checkpoints/_.ckpt', help='Path to model checkpoint')
    parser.add_argument('--dataset_root', type=str,required=True,
                        default='/home/perception/Datasets/nuscenes/',
                        help='Path to dataset root')
    parser.add_argument('--nusc_version',type=str, default='v1.0-trainval',
                        help='Nuscenes dataset version')
    parser.add_argument('--seq_id',help='Sequence id to visualize,' + \
                        'if not specified all val dataset will be used',
                        type=int, default=-1)
    parser.add_argument('--save_path', help='Path to save results',
                        type=str, default='results')
    parser.add_argument('--save_images', type=bool, default=False,
                        help='Save multi-camera input images along with the results')
    args = parser.parse_args()



    main(args)
