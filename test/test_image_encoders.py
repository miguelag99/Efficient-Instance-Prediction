import os
import numpy as np
import pytest
import sys
import torch
 
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(root_dir)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
 
@pytest.mark.parametrize("feature_extractor", ['resnet18', 'resnet50', 'resnet101'])
@pytest.mark.parametrize("out_channels", [64])
@pytest.mark.parametrize("depth_channels", [48])
@pytest.mark.parametrize("downsample_factor", [4, 8, 16, 32])
@torch.inference_mode()
def test_resnetfpn(feature_extractor, out_channels, depth_channels, downsample_factor):
 
    from prediction.model.resnet_encoder import ResNetFPNEncoder
 
    model = ResNetFPNEncoder(
        feature_extractor=feature_extractor,
        out_channels=out_channels,
        depth_channels=depth_channels,
        downsample=downsample_factor,
        depth_distribution=True,
    ).to(device)
 
    x = torch.randn(2, 3, 224, 480).to(device)
    x = model(x)
    assert x.shape[1] == out_channels
    assert x.shape[2] == depth_channels
    assert x.shape[-2] == 224 // downsample_factor
    assert x.shape[-1] == 480 // downsample_factor
    
@pytest.mark.parametrize("feature_extractor", ['efficientnet-b0', 'efficientnet-b4'])
@pytest.mark.parametrize("out_channels", [64])
@pytest.mark.parametrize("depth_channels", [48])
@pytest.mark.parametrize("downsample_factor", [8, 16])
@torch.inference_mode()
def test_efficientnet(feature_extractor, out_channels, depth_channels, downsample_factor):
 
    from prediction.model.efficientnet_encoder import EncoderEfficientNet
 
    model = EncoderEfficientNet(out_channels=out_channels,
                                depth_distribution=True,
                                depth_channels=depth_channels,
                                downsample=downsample_factor,
                                model_name=feature_extractor,
                                ).to(device)
 
    x = torch.randn(36, 3, 224, 480).to(device)
    x = model(x)
    assert x.shape[1] == out_channels
    assert x.shape[2] == depth_channels
    assert x.shape[-2] == 224 // downsample_factor
    assert x.shape[-1] == 480 // downsample_factor
    
