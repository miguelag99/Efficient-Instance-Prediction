import matplotlib.pyplot as plt
import torch
from einops import rearrange
from matplotlib.patches import Circle
from tqdm import tqdm

import sys
sys.path.append('..')

from prediction.config import get_config
from prediction.data.prepare_loader import prepare_dataloaders

def get_reference_points(H, W, Z=8, num_points_in_pillar=4, dim='3d', bs=1,
                         device='cuda', dtype=torch.float):
        """Get the reference points used in SCA and TSA.
        Args:
            H, W: spatial shape of bev.
            Z: hight of pillar.
            D: sample D points uniformly from each pillar.
            device (obj:`device`): The device where
                reference_points should be.
        Returns:
            Tensor: reference points used in decoder, has \
                shape (bs, num_keys, num_levels, 2).
        """

        # reference points in 3D space, used in spatial cross-attention (SCA)
        if dim == '3d':
            zs = torch.linspace(0.5, Z - 0.5, num_points_in_pillar, dtype=dtype,
                                device=device).view(-1, 1, 1).expand(
                                    num_points_in_pillar, H, W) / Z
            xs = torch.linspace(0.5, W - 0.5, W, dtype=dtype,
                                device=device).view(1, 1, W).expand(
                                    num_points_in_pillar, H, W) / W
            ys = torch.linspace(0.5, H - 0.5, H, dtype=dtype,
                                device=device).view(1, H, 1).expand(
                                    num_points_in_pillar, H, W) / H
            ref_3d = torch.stack((xs, ys, zs), -1)
            ref_3d = ref_3d.permute(0, 3, 1, 2).flatten(2).permute(0, 2, 1)
            ref_3d = ref_3d[None].repeat(bs, 1, 1, 1)
            return ref_3d

        # reference points on 2D bev plane, used in temporal self-attention (TSA).
        elif dim == '2d':
            ref_y, ref_x = torch.meshgrid(
                torch.linspace(
                    0.5, H - 0.5, H, dtype=dtype, device=device),
                torch.linspace(
                    0.5, W - 0.5, W, dtype=dtype, device=device),
                indexing='ij'
            )
            ref_y = ref_y.reshape(-1)[None] / H
            ref_x = ref_x.reshape(-1)[None] / W
            ref_2d = torch.stack((ref_x, ref_y), -1)
            ref_2d = ref_2d.repeat(bs, 1, 1).unsqueeze(2)
            return ref_2d

if __name__ == '__main__':
    
        
    cfg, hparams = get_config(cfg_file='bevformer_tiny_short')

    
    
    cfg.DATASET.VERSION = "v1.0-mini"
    cfg.N_WORKERS = 0
    cfg.BATCHSIZE = 1

    _ , valloader = prepare_dataloaders(cfg, return_dataset=False,
                                        return_orig_images=True, return_pcl=True)
    
    receptive_field = 3
    
    data = next(iter(valloader))
    image = data['image'][:, :receptive_field].contiguous()
    full_image = data['orig_image'][:, :receptive_field].contiguous()
    intrinsics = data['orig_intrinsics'][:, :receptive_field].contiguous()
    extrinsics = data['extrinsics'][:, :receptive_field].contiguous()
    future_egomotion = data['future_egomotion'][:, :receptive_field].contiguous()
    lidar_ego_2_img = data['lidar_ego_2_img'][:, :receptive_field].contiguous()
    
    B, S, N, _, _, _= image.shape      
    
    pc_range = [-50.0, -50.0, -10.0, 50.0, 50.0, 10.0]
    reference_points = get_reference_points(
        200,200,pc_range[5]-pc_range[2],
        num_points_in_pillar=1,
        dim='3d',
        bs=1,
        device='cuda',
        dtype=torch.float32
    ).cpu()
    
    reference_points[..., 0:1] = reference_points[..., 0:1] * \
        (pc_range[3] - pc_range[0]) + pc_range[0]
    reference_points[..., 1:2] = reference_points[..., 1:2] * \
        (pc_range[4] - pc_range[1]) + pc_range[1]
    reference_points[..., 2:3] = reference_points[..., 2:3] * \
        (pc_range[5] - pc_range[2]) + pc_range[2]

    reference_points = torch.cat(
        (reference_points, torch.ones_like(reference_points[..., :1])), -1)
    reference_points = rearrange(reference_points, 'b n d c -> b c (n d)')
        
    reference_points = reference_points[:,:, (reference_points[:,0,:]>0).squeeze()]

    IMG_HEIGHT, IMG_WIDTH = full_image.shape[-2:]
    fig, ax = plt.subplots()
    ax.imshow(full_image[0,0,1].permute(1,2,0).cpu().numpy())
        
    # Only front camera from frame 0
    projected_reference_point = lidar_ego_2_img[:,0,1,...] @ reference_points
    projected_reference_point[:,:3,:] = projected_reference_point[:,:3,:] / projected_reference_point[:,2,:]

    pt_filter = torch.logical_and(projected_reference_point[:,0] > 0,
                                  projected_reference_point[:,1] > 0)
    pt_filter = torch.logical_and(pt_filter,
                                  projected_reference_point[:,1] < IMG_HEIGHT)
    pt_filter = torch.logical_and(pt_filter,
                                  projected_reference_point[:,0] < IMG_WIDTH).squeeze()
    
    projected_reference_point = projected_reference_point[:,:,pt_filter]
    
    if projected_reference_point.shape[-1] > 0:    
        for i in tqdm(range(projected_reference_point.shape[-1])):
            ax.add_patch(Circle((projected_reference_point[0,0,i],
                                projected_reference_point[0,1,i]), 1, color='r'))


    plt.savefig('3d_anchors_image_1p.png')


        
