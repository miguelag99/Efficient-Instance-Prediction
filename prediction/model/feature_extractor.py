import torch
import torch.nn as nn

from prediction.model.camera_encoder import Encoder
from prediction.utils.network import pack_sequence_dim, unpack_sequence_dim
from prediction.utils.geometry import VoxelsSumming, calculate_birds_eye_view_parameters, cumulative_warp_features

class FeatureExtractor(nn.Module):
    def __init__(self,
                 x_bound = (-50.0, 50.0, 0.5),  # Forward
                 y_bound = (-50.0, 50.0, 0.5),  # Sides
                 z_bound = (-10.0, 10.0, 20.0),  # Height
                 d_bound = (2.0, 50.0, 1.0),
                 downsample: int = 8,
                 out_channels: int = 64,
                 receptive_field: int = 3,
                 pred_frames: int = 4,
                 latent_dim: int = 32,
                 use_depth_distribution: bool = True,
                 model_name: str = 'efficientnet-b0',
                 img_size = (224,480)
                 ):
        super().__init__()

        self.bounds = {
            'x': x_bound,
            'y': y_bound,
            'z': z_bound,
            'd': d_bound
        }

        bev_resolution, bev_start_position, bev_dimension = calculate_birds_eye_view_parameters(
            x_bound, y_bound, z_bound)
        self.bev_resolution = nn.Parameter(bev_resolution, requires_grad=False)
        self.bev_start_position = nn.Parameter(bev_start_position, requires_grad=False)
        self.bev_dimension = nn.Parameter(bev_dimension, requires_grad=False)

        self.img_final_dim = img_size

        self.encoder_downsample = downsample
        self.encoder_out_channels = out_channels

        self.frustum = self.create_frustum()
        self.depth_channels, _, _, _ = self.frustum.shape

        # temporal block
        self.receptive_field = receptive_field
        self.n_future = pred_frames
        # self.latent_dim = latent_dim

        # Spatial extent in bird's-eye view, in meters
        self.spatial_extent = (x_bound[1], y_bound[1])
        self.bev_size = (self.bev_dimension[0].item(), self.bev_dimension[1].item())

        # Define the camera multi-sweep encoder
        self.encoder = Encoder(out_channels=out_channels,
                               depth_distribution=use_depth_distribution,
                               depth_channels=self.depth_channels,
                               downsample=downsample,
                               model_name=model_name,
                               )

    def forward(self, image, intrinsics, extrinsics, future_egomotion):
        '''
        Inputs:
            image: (b, sweeps, n_cam, channel, h, w)
            intrinsic: (b, sweeps, n_cam, 3, 3)
            extrinsic: (b, sweeps, n_cam, 4, 4)
            future_egomotion: (b, sweeps, 6)
        '''
        # Only process features from the past and present (within receptive field)
        image = image[:, :self.receptive_field].contiguous()
        intrinsics = intrinsics[:, :self.receptive_field].contiguous()
        extrinsics = extrinsics[:, :self.receptive_field].contiguous()
        future_egomotion = future_egomotion[:, :self.receptive_field].contiguous()

        x = self.bev_features(image, extrinsics, intrinsics)

        # Warp past features to the present's reference frame
        x = cumulative_warp_features(
            x.clone(), future_egomotion,
            mode='bilinear', spatial_extent=self.spatial_extent,
        )

        return x

    def create_frustum(self) -> nn.Parameter:
        # Create grid in image plane
        h, w = self.img_final_dim
        downsampled_h, downsampled_w = h // self.encoder_downsample, w // self.encoder_downsample

        # Depth grid
        depth_grid = torch.arange(*self.bounds['d'], dtype=torch.float)
        depth_grid = depth_grid.view(-1, 1, 1).expand(-1, downsampled_h, downsampled_w)
        n_depth_slices = depth_grid.shape[0]

        # x and y grids
        x_grid = torch.linspace(0, w - 1, downsampled_w, dtype=torch.float)
        x_grid = x_grid.view(1, 1, downsampled_w).expand(n_depth_slices, downsampled_h, downsampled_w)
        y_grid = torch.linspace(0, h - 1, downsampled_h, dtype=torch.float)
        y_grid = y_grid.view(1, downsampled_h, 1).expand(n_depth_slices, downsampled_h, downsampled_w)

        # Dimension (n_depth_slices, downsampled_h, downsampled_w, 3)
        # containing data points in the image: left-right, top-bottom, depth
        frustum = torch.stack((x_grid, y_grid, depth_grid), -1)
        return nn.Parameter(frustum, requires_grad=False)
    
    def bev_features(self, x, extrinsics, intrinsics):
        '''
        Inputs:
            x: (b, sweeps, n_cam, channel, h, w)
            extrinsics: (b, sweeps, n_cam, 4, 4)
            intrinsics: (b, sweeps, n_cam, 3, 3)
        '''
        batch, sweeps, n_cam, channel, h, w = x.shape

        # Reshape to (b*sweeps*n_cam, channel, h, w)
        x = pack_sequence_dim(x)
        intrinsics = pack_sequence_dim(intrinsics)
        extrinsics = pack_sequence_dim(extrinsics) 

        geometry = self.get_geometry(intrinsics, extrinsics)
        x = self.encoder_forward(x)
        x = self.projection_to_birds_eye_view(x, geometry)
        x = unpack_sequence_dim(x, batch, sweeps)
        return x
   
    def get_geometry(self, intrinsics, extrinsics):
        """Calculate the (x, y, z) 3D position of the features.
        """
        rotation, translation = extrinsics[..., :3, :3], extrinsics[..., :3, 3]
        B, N, _ = translation.shape
        # Add batch, camera dimension, and a dummy dimension at the end
        points = self.frustum.unsqueeze(0).unsqueeze(0).unsqueeze(-1)

        # Camera to ego reference frame
        points = torch.cat((points[:, :, :, :, :, :2] * points[:, :, :, :, :, 2:3], points[:, :, :, :, :, 2:3]), 5)
        combined_transformation = rotation.matmul(torch.inverse(intrinsics))
        points = combined_transformation.view(B, N, 1, 1, 1, 3, 3).matmul(points).squeeze(-1)
        points += translation.view(B, N, 1, 1, 1, 3)

        # The 3 dimensions in the ego reference frame are: (forward, sides, height)
        return points

    def encoder_forward(self, x):
        # batch, n_cameras, channels, height, width
        b, n, c, h, w = x.shape

        x = x.view(b * n, c, h, w)
        x = self.encoder(x)
        x = x.view(b, n, *x.shape[1:])
        x = x.permute(0, 1, 3, 4, 5, 2)

        return x

    def projection_to_birds_eye_view(self, x, geometry):
        """ Adapted from https://github.com/nv-tlabs/lift-splat-shoot/blob/master/src/models.py#L200"""
        # batch, n_cameras, depth, height, width, channels
        batch, n, d, h, w, c = x.shape
        output = torch.zeros(
            (batch, c, self.bev_dimension[0], self.bev_dimension[1]), dtype=torch.float, device=x.device
        )

        # Number of 3D points
        N = n * d * h * w
        for b in range(batch):
            # flatten x
            x_b = x[b].reshape(N, c)

            # Convert positions to integer indices
            geometry_b = ((geometry[b] - (self.bev_start_position - self.bev_resolution / 2.0)) / self.bev_resolution)
            geometry_b = geometry_b.view(N, 3).long()

            # Mask out points that are outside the considered spatial extent.
            mask = (
                    (geometry_b[:, 0] >= 0)
                    & (geometry_b[:, 0] < self.bev_dimension[0])
                    & (geometry_b[:, 1] >= 0)
                    & (geometry_b[:, 1] < self.bev_dimension[1])
                    & (geometry_b[:, 2] >= 0)
                    & (geometry_b[:, 2] < self.bev_dimension[2])
            )
            x_b = x_b[mask]
            geometry_b = geometry_b[mask]

            # Sort tensors so that those within the same voxel are consecutives.
            ranks = (
                    geometry_b[:, 0] * (self.bev_dimension[1] * self.bev_dimension[2])
                    + geometry_b[:, 1] * (self.bev_dimension[2])
                    + geometry_b[:, 2]
            )
            ranks_indices = ranks.argsort()
            x_b, geometry_b, ranks = x_b[ranks_indices], geometry_b[ranks_indices], ranks[ranks_indices]

            # Project to bird's-eye view by summing voxels.
            x_b, geometry_b = VoxelsSumming.apply(x_b, geometry_b, ranks)

            bev_feature = torch.zeros((self.bev_dimension[2], self.bev_dimension[0], self.bev_dimension[1], c),
                                      device=x_b.device)
            bev_feature[geometry_b[:, 2], geometry_b[:, 0], geometry_b[:, 1]] = x_b

            # Put channel in second position and remove z dimension
            bev_feature = bev_feature.permute((0, 3, 1, 2))
            bev_feature = bev_feature.squeeze(0)

            output[b] = bev_feature

        return output
