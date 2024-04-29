import os
import socket
import time
import argparse
import numpy as np
import random

import lightning as L
import torch
import yaml

from prediction.data.prepare_loader import prepare_dataloaders
from prediction.configs import baseline_cfg
from prediction.config import namespace_to_dict
from prediction.trainer import TrainingModule

def main(args):
    
    # Load base config

    bcfg = baseline_cfg
        
    ckpt = args.checkpoint
    assert os.path.exists(ckpt),\
        f'Checkpoint {ckpt} does not exist'
        
    l_module = TrainingModule.load_from_checkpoint(ckpt)
    
    cfg = l_module.cfg
    cfg.DATASET.DATAROOT = args.dataset_root
    cfg.DATASET.VERSION = args.nusc_version
    cfg.BATCHSIZE = bcfg.BATCHSIZE
    cfg.N_WORKERS = bcfg.N_WORKERS
        
    # Set random seed for reproducibility
    seed = 42
    L.seed_everything(seed, workers=True)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    _ , valloader = prepare_dataloaders(cfg)
                 
           
    import pdb; pdb.set_trace()
    
    l_module.eval()

    trainer = L.Trainer(
        accelerator=cfg.ACCELERATOR,
        devices=cfg.DEVICES,
        precision=cfg.PRECISION,
        sync_batchnorm=True,
        gradient_clip_val=cfg.GRAD_NORM_CLIP,
        max_epochs=cfg.EPOCHS,
        log_every_n_steps=cfg.LOGGING_INTERVAL,
    )
    
    metrics = trainer.validate(l_module, valloader)
            
    # Free memory
    del l_module
    del trainer
    del valloader
    torch.cuda.empty_cache()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str,required=True, help='Path to model checkpoint')
    parser.add_argument('--dataset_root', type=str, default='/home/perception/Datasets/nuscenes', help='Path to dataset root')
    parser.add_argument('--nusc_version',type=str, default='v1.0-trainval', help='Nuscenes dataset version')
    args = parser.parse_args()

    main(args)