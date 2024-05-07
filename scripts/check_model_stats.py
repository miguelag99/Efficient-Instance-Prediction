import argparse
import numpy as np
import random

import lightning as L
import torch

from thop import profile

import sys
sys.path.append('../')

from prediction.configs import baseline_cfg
from prediction.config import namespace_to_dict
from prediction.models.powerbev import PowerBEV
from prediction.powerformer.predictor import FullSegformerCustomHead
from prediction.trainer import TrainingModule




EVAL_POWERBEV = True
EVAL_FULLSEGFORMER = False


def main(args):

    # Load training config
    if args.config == 'baseline':
        cfg = baseline_cfg
        
    hparams = namespace_to_dict(cfg)

    # Set random seed for reproducibility
    seed = 42
    L.seed_everything(seed, workers=True)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    t1 = torch.cuda.Event(enable_timing=True)
    t2 = torch.cuda.Event(enable_timing=True)
    

    # Image: [1,7,6,3,224,480]
    # Intrinsics: [1,7,6,3,3]
    # Extrinsics: [1,7,6,4,4]
    # Future egomotion: [1,7,6]
    # Future distribution inputs: [1,6,6,200,200]
    
    bs = 1
    batch = {
        'image': torch.rand(bs,7,6,3,224,480).cuda(),
        'intrinsics': torch.rand(bs,7,6,3,3).cuda(),
        'extrinsics': torch.rand(bs,7,6,4,4).cuda(),
        'future_egomotion': torch.rand(bs,7,6).cuda(),
        'future_distribution_inputs': torch.rand(bs,6,6,200,200).cuda()
    }
    
    
    # PowerBEV measures
    if EVAL_POWERBEV:
        
        gpu_init_mem = torch.cuda.max_memory_allocated(device=None)
        print(f"GPU init mem: {gpu_init_mem/(1024*1024)} MB")
                
        ## PowerBEV measures
        model = PowerBEV(cfg).to('cuda')
        model(batch['image'], batch['intrinsics'], batch['extrinsics'], batch['future_egomotion'])

        model.eval()   
        t = 0
        tv = []
        for i in range(20):
            t1.record()
            model(batch['image'], batch['intrinsics'], batch['extrinsics'], batch['future_egomotion'])
            t2.record()
            torch.cuda.synchronize()
            t += t1.elapsed_time(t2)
            tv.append(int(t1.elapsed_time(t2)))
        
        print('\033[91m PowerBEV took {} ms \033[0m'.format(t/20))
        print(tv)
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Number of parameters: {total_params/1000000}M")
        used_vram = torch.cuda.max_memory_allocated(device=None) - gpu_init_mem
        print(f"GPU mem usage: {used_vram/(1024*1024)} MB\n")
        
        flops, _ = profile(model, inputs=(batch['image'], batch['intrinsics'], batch['extrinsics'], batch['future_egomotion'], None))
        print('{:.2f} G \tTotal FLOPs'.format(flops/1000**3))
                
        from fvcore.nn import FlopCountAnalysis, flop_count_table
        model.to('cuda')
        flops = FlopCountAnalysis(model, (batch['image'], batch['intrinsics'], batch['extrinsics'], batch['future_egomotion'], None))
        print(flop_count_table(flops))
        flops = flops.total()
        print('{:.2f} G \tTotal FLOPs'.format(flops/1000**3))
        
        del model
        torch.cuda.reset_peak_memory_stats()
        print('---------------------------------')
        
        exit()
       
       
    
    ## FullSegformer measures
    if EVAL_FULLSEGFORMER:
        
        gpu_init_mem = torch.cuda.max_memory_allocated(device=None)
        print(f"GPU init mem: {gpu_init_mem/(1024*1024)} MB")
        
        model = FullSegformerCustomHead(cfg).to('cuda')
        model.eval()
        out = model(batch['image'], batch['intrinsics'], batch['extrinsics'], batch['future_egomotion'])
       
        model.eval()   
        t = 0
        tv = []
        for i in range(20):
            t1.record()
            model(batch['image'], batch['intrinsics'], batch['extrinsics'], batch['future_egomotion'])
            t2.record()
            torch.cuda.synchronize()
            t += t1.elapsed_time(t2)
            tv.append(int(t1.elapsed_time(t2)))
        
        print('\033[91m PowerBEV took {} ms \033[0m'.format(t/20))
        print(tv)
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Number of parameters: {total_params/1000000}M")
        used_vram = torch.cuda.max_memory_allocated(device=None) - gpu_init_mem
        print(f"GPU mem usage: {used_vram/(1024*1024)} MB\n")
        
        flops, _ = profile(model, inputs=(batch['image'], batch['intrinsics'], batch['extrinsics'], batch['future_egomotion'], None))
        print('{:.2f} G \tTotal FLOPs'.format(flops/1000**3))
        
        from fvcore.nn import FlopCountAnalysis, flop_count_table
        model.to('cuda')
        flops = FlopCountAnalysis(model, (batch['image'], batch['intrinsics'], batch['extrinsics'], batch['future_egomotion'], None))
        print(flop_count_table(flops))
        flops = flops.total()
        print('{:.2f} G \tTotal FLOPs'.format(flops/1000**3))
               
        del model
        torch.cuda.reset_peak_memory_stats()
        print('---------------------------------')
        
        exit()

if __name__ == "__main__":
    # Create parser with one argument
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='baseline')
    args = parser.parse_args()


    main(args)