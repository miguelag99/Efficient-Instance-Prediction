from typing import Literal

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, resnet50, resnet101
from torchvision.models.feature_extraction import create_feature_extractor

class TopDownBlock(nn.Module):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        use_act: bool = True,
        use_merge: bool = True,
        ) -> None:
        super().__init__()

        self.use_merge = use_merge
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest') if \
            use_merge else nn.Identity()
        self.pre_merge = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.conv = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.act_fn = nn.SiLU() if use_act else nn.Identity()

    def forward(self, x1, x2=None):
        """x1: bottom-up pathway, x2: top-down pathway.
        """
        x2 = self.upsample(x2)
        x1 = self.pre_merge(x1)
        if self.use_merge:
            x = self.conv(self.act_fn(x1 + x2))
        else:
            x = self.conv(self.act_fn(x1))
        return x

class ResNetFPN(nn.Module):

    def __init__(
        self,
        feature_extractor: Literal['resnet18', 'resnet50', 'resnet101'],
        out_channels: int = 256,
    ) -> None:
        super().__init__()

        assert feature_extractor in ['resnet18', 'resnet50', 'resnet101']
                     
        match feature_extractor:
            case 'resnet18':
                model = resnet18(weights="IMAGENET1K_V1")
            case 'resnet50':
                model = resnet50(weights="IMAGENET1K_V2")
            case 'resnet101':
                model = resnet101(weights="IMAGENET1K_V2")
                
        self.backbone = create_feature_extractor(
            model,
            return_nodes=['layer1', 'layer2', 'layer3', 'layer4'])
        conv_out_channels = self._get_backbone_out_channels()
        
        self.top_down_block5 = TopDownBlock(conv_out_channels[-1], out_channels,
                                            use_merge=False)
        self.top_down_block4 = TopDownBlock(conv_out_channels[-2], out_channels)
        self.top_down_block3 = TopDownBlock(conv_out_channels[-3], out_channels)
        self.top_down_block2 = TopDownBlock(conv_out_channels[-4], out_channels)

        
    def _get_backbone_out_channels(self):
        inp = torch.randn(1, 3, 224, 224)
        out_channels = []
        for _, v in self.backbone(inp).items():
            out_channels.append(v.shape[1])
        return out_channels

    def forward(self, x):

        # Bottom-up pathway.
        features = self.backbone(x)
        c2, c3, c4, c5 = features['layer1'], features['layer2'], \
            features['layer3'], features['layer4']

        # Top-down pathway with lateral connections.
        p5 = self.top_down_block5(c5, None)
        p4 = self.top_down_block4(c4, p5)
        p3 = self.top_down_block3(c3, p4)
        p2 = self.top_down_block2(c2, p3)

        return p2, p3, p4, p5
    
class ResNetFPNEncoder(nn.Module):

    def __init__(
        self,
        feature_extractor: Literal['resnet18', 'resnet50', 'resnet101'],
        out_channels: int = 64,
        depth_channels: int = 32,
        downsample: int = 8,
        depth_distribution: bool = True,
        return_depth_map: bool = False,
    ) -> None:
        super().__init__()

        self.use_depth_distribution = depth_distribution
        self.return_depth_map = return_depth_map
        
        if depth_distribution:
            self.D = depth_channels
            self.C = out_channels + depth_channels
        else:
            self.C = out_channels
        
        assert self.C > 0

        self.fpn = ResNetFPN(feature_extractor, out_channels = self.C)
        if downsample not in [4, 8, 16, 32]:
            raise ValueError("Downsample must be one of [4, 8, 16, 32]")
        self.downsample = downsample

    def forward(self, x : torch.Tensor) -> torch.Tensor:
        
        B,C,H,W = x.shape
        
        p2, p3, p4, p5 = self.fpn(x)
                
        if self.downsample == 4:
            x = p2
        elif self.downsample == 8:
            x = p3
        elif self.downsample == 16:
            x = p4
        elif self.downsample == 32:
            x = p5
        else:
            raise ValueError("Downsample must be one of [4, 8, 16, 32]")
        
        if self.use_depth_distribution:
            depth_channels = F.softmax(x[:, :self.D, ...], dim=1)
            context_channels = x[:, self.D:, ...]
            x = depth_channels.unsqueeze(1) * context_channels.unsqueeze(2)
            if self.return_depth_map:
                return x, torch.argmax(depth_channels,dim=1)
        return x, None
    
    def _pack_seq_dim(self, x: torch.Tensor) -> torch.Tensor:
        # Pack the seq and multicamera dimension.
        B, S, N = x.shape[:3]

        return x.view(B*S*N,*x.shape[3:])
    
    def _unpack_seq_dim(self, x: torch.Tensor,
                        B: int, S: int, N: int) -> torch.Tensor:
        # Unpack the seq and multicamera dimension.
        return x.view(B, S, N, *x.shape[1:])
    
    
if __name__ == '__main__':
    
    # Simple inference test
    
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
            
    # Create input data
    img = torch.randn(7, 3, 224, 480).to(device)
    b, c, h, w = img.shape

    ## Reshape to be divisible by 32
    r1 = img.shape[-2] % 32
    r2 = img.shape[-1] % 32
    img= img[...,:img.shape[-2] - r1,:img.shape[-1] - r2]
    
    # Define the models
    feature_extractor = ResNetFPNEncoder('resnet18', out_channels = 64,
                            depth_channels = 48, downsample = 8,
                            depth_distribution = True).to(device)
    
    with torch.inference_mode():
        feats = feature_extractor(img)
        print(feats.shape) # Expected output [7, 64, 48, 28, 60]
