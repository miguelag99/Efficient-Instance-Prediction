import torch
import torch.nn as nn
import numpy as np
import time

from transformers import SegformerModel, SegformerConfig, SegformerForSemanticSegmentation

from prediction.model.feature_extractor import FeatureExtractor
from prediction.layers.spatial_temporal import Residual

    
class FullSegformerCustomHead(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        self.feature_width = int((self.cfg.LIFT.X_BOUND[1] - self.cfg.LIFT.X_BOUND[0])\
                              /self.cfg.LIFT.X_BOUND[2])

        self.use_ego_motion = self.cfg.MODEL.STCONV.INPUT_EGOPOSE
        self.receptive_field = self.cfg.TIME_RECEPTIVE_FIELD

        self.use_depth_distribution = self.cfg.MODEL.ENCODER.USE_DEPTH_DISTRIBUTION
        self.temporal_attn_channels = self.cfg.MODEL.ENCODER.OUT_CHANNELS
        if self.use_ego_motion:
            self.temporal_attn_channels += 6
        self.lidar_supervision = self.cfg.LIDAR_SUPERVISION

        self.feature_extractor = FeatureExtractor(
                 x_bound = self.cfg.LIFT.X_BOUND,
                 y_bound = self.cfg.LIFT.Y_BOUND,
                 z_bound = self.cfg.LIFT.Z_BOUND,
                 d_bound = self.cfg.LIFT.D_BOUND,
                 downsample = self.cfg.MODEL.ENCODER.DOWNSAMPLE,
                 out_channels = self.cfg.MODEL.ENCODER.OUT_CHANNELS,
                 receptive_field = self.cfg.TIME_RECEPTIVE_FIELD,
                 pred_frames = self.cfg.N_FUTURE_FRAMES,
                 latent_dim = self.cfg.MODEL.DISTRIBUTION.LATENT_DIM,
                 use_depth_distribution = self.cfg.MODEL.ENCODER.USE_DEPTH_DISTRIBUTION,
                 model_name = self.cfg.MODEL.ENCODER.NAME,
                 img_size = self.cfg.IMAGE.FINAL_DIM,
                 return_depth_map=self.lidar_supervision
                 )
                      
        segformer_out_dim = len(self.cfg.SEMANTIC_SEG.WEIGHTS)*(self.cfg.N_FUTURE_FRAMES + 2)*self.cfg.MODEL.SEGFORMER.HEAD_DIM_MULTIPLIER
       
        segformer_config = SegformerConfig(
            image_size = self.feature_width,
            num_channels = self.cfg.TIME_RECEPTIVE_FIELD*self.temporal_attn_channels,
            num_encoder_blocks = self.cfg.MODEL.SEGFORMER.N_ENCODER_BLOCKS,
            depths = self.cfg.MODEL.SEGFORMER.DEPTHS,
            sr_ratios = self.cfg.MODEL.SEGFORMER.SEQUENCE_REDUCTION_RATIOS,
            hidden_sizes = self.cfg.MODEL.SEGFORMER.HIDDEN_SIZES, # No receptive field multiplication
            patch_sizes = self.cfg.MODEL.SEGFORMER.PATCH_SIZES,
            strides = self.cfg.MODEL.SEGFORMER.STRIDES,
            num_attention_heads = self.cfg.MODEL.SEGFORMER.NUM_ATTENTION_HEADS,
            mlp_ratios = self.cfg.MODEL.SEGFORMER.MLP_RATIOS,
            output_hidden_states = True,
            return_dict = True,
            num_labels = segformer_out_dim
        )
        
        kernel = self.cfg.MODEL.SEGFORMER.HEAD_KERNEL
        stride = self.cfg.MODEL.SEGFORMER.HEAD_STRIDE
        segformer_out_dim = 256
        
        # Instantiate the two different banches and change the classifier layer with our head
        self.segmentation_branch = SegformerForSemanticSegmentation(segformer_config)
        self.segmentation_branch.decode_head.classifier = nn.Sequential(
            Residual(segformer_out_dim, segformer_out_dim//2),
            Residual(segformer_out_dim//2, segformer_out_dim//2),
            Residual(segformer_out_dim//2, segformer_out_dim//4),
            Residual(segformer_out_dim//4, segformer_out_dim//4),
            nn.ConvTranspose2d(segformer_out_dim//4,
                               len(self.cfg.SEMANTIC_SEG.WEIGHTS)*(self.cfg.N_FUTURE_FRAMES + 2),
                               kernel_size=kernel, stride=stride)
        )
 
        # segformer_config.num_labels = 2
        self.flow_branch = SegformerForSemanticSegmentation(segformer_config)
        self.flow_branch.decode_head.classifier = nn.Sequential(
            Residual(segformer_out_dim, segformer_out_dim//2),
            Residual(segformer_out_dim//2, segformer_out_dim//2),
            Residual(segformer_out_dim//2, segformer_out_dim//4),
            Residual(segformer_out_dim//4, segformer_out_dim//4),
            nn.ConvTranspose2d(segformer_out_dim//4, 2*(self.cfg.N_FUTURE_FRAMES + 2),
                               kernel_size=kernel, stride=stride)
        )


        
    def forward(self, x, intrinsics, extrinsics, future_egomotion,  future_distribution_inputs=None, noise=None):
        output = {}
        start_time = time.time()

        # Image feature extraction
        x, depth_maps = self.feature_extractor(x, 
                                               intrinsics, extrinsics, 
                                               future_egomotion)

        if self.lidar_supervision:
            output['depth_maps'] = depth_maps
        
        perception_time = time.time()

        # Transofrmer multi-scale encoder
        b, s, c = future_egomotion.shape
        h, w = x.shape[-2:]
        future_egomotions_spatial = future_egomotion.view(b, s, c, 1, 1).expand(b, s, c, h, w)
        # At time 0, no egomotion so feed zero vector
        future_egomotions_spatial = torch.cat([torch.zeros_like(future_egomotions_spatial[:, :1]),
                                            future_egomotions_spatial[:, :(self.receptive_field-1)]], dim=1)
        x = torch.cat([x, future_egomotions_spatial], dim=-3)

        b, t, c, h, w = x.shape
        x = x.view(b, t*c, h, w)

        # Segformer directly
        seg_out= self.segmentation_branch(x).logits
        flow_out = self.flow_branch(x).logits
               
        # seg_out= self.conv_seg(seg_out)
        # flow_out = self.conv_flow(flow_out)      
                                
        output['segmentation'] = seg_out.view(b,self.cfg.N_FUTURE_FRAMES + 2,
                                              len(self.cfg.SEMANTIC_SEG.WEIGHTS), h, w).contiguous()
        output['instance_flow'] = flow_out.view(b,self.cfg.N_FUTURE_FRAMES + 2,
                                       len(self.cfg.SEMANTIC_SEG.WEIGHTS), h, w).contiguous()
        
        prediction_time = time.time()

        output['perception_time'] = perception_time - start_time
        output['prediction_time'] = prediction_time - perception_time
        output['total_time'] = output['perception_time'] + output['prediction_time']

        output = {**output}

        return output
