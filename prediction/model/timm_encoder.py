from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

import timm

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

class TimmFPN(nn.Module):

    def __init__(
        self,
        feature_extractor: str = 'mobilenetv4_conv_small.e2400_r224_in1k',
        out_channels: int = 256,
    ) -> None:
        super().__init__()
        
        self.backbone = timm.create_model(
            feature_extractor,
            pretrained=True,
            features_only=True)
        
        conv_out_channels = self._get_backbone_out_channels()
        
        self.top_down_block5 = TopDownBlock(conv_out_channels[-1], out_channels,
                                            use_merge=False)
        self.top_down_block4 = TopDownBlock(conv_out_channels[-2], out_channels)
        self.top_down_block3 = TopDownBlock(conv_out_channels[-3], out_channels)
        self.top_down_block2 = TopDownBlock(conv_out_channels[-4], out_channels)

        
    def _get_backbone_out_channels(self):
        inp = torch.randn(1, 3, 224, 224)
        out_channels = []
        for v in self.backbone(inp):
            out_channels.append(v.shape[1])
        return out_channels

    def forward(self, x):

        # Bottom-up pathway.
        features = self.backbone(x)
        c2, c3, c4, c5 = features[-4], features[-3], \
            features[-2], features[-1]

        # Top-down pathway with lateral connections.
        p5 = self.top_down_block5(c5, None)
        p4 = self.top_down_block4(c4, p5)
        p3 = self.top_down_block3(c3, p4)
        p2 = self.top_down_block2(c2, p3)

        return p2, p3, p4, p5
    
class TimmFPNEncoder(nn.Module):

    def __init__(
        self,
        feature_extractor: str  = 'mobilenetv4_conv_small.e2400_r224_in1k',
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

        self.fpn = TimmFPN(feature_extractor, out_channels = self.C)
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
    model = TimmFPNEncoder(out_channels=64, depth_channels=48,
                           downsample=8, depth_distribution=True).to(device)

    with torch.inference_mode():
        feats = model(img)
        print(feats.shape) # Expected output [7, 64, 48, 28, 60]
