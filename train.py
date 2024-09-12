import os
import socket
import warnings
import time
import argparse
import importlib
import numpy as np
import random

import lightning as L
import torch

from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint, ModelSummary, LearningRateMonitor

from prediction.data.prepare_loader import prepare_dataloaders
from prediction.config import get_config
from prediction.trainer import TrainingModule


def main(args):
    
    cfg, hparams = get_config(cfg_dir='./prediction/configs/', 
                     cfg_file=args.config)
    
    if cfg.PRETRAINED.RESUME_TRAINING:
        save_dir = cfg.PRETRAINED.PATH
        if cfg.WANDB_ID == '':
            warnings.warn("Wandb ID not provided. Logging will start a new run.")
            wdb_logger = WandbLogger(project=cfg.WANDB_PROJECT,save_dir=save_dir,
                    log_model=False, name=cfg.TAG)
        else:   
            wdb_logger = WandbLogger(project=cfg.WANDB_PROJECT,save_dir=save_dir,
                                    log_model=False, name=cfg.TAG,
                                    id=cfg.WANDB_ID, resume='must')
    else:
        save_dir = os.path.join(
            cfg.LOG_DIR, time.strftime('%d%b%Yat%H:%M') + '_' + socket.gethostname() + '_' + cfg.TAG
        ) 
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
            os.makedirs(os.path.join(save_dir, 'checkpoints'))
            wdb_logger = WandbLogger(project=cfg.WANDB_PROJECT,save_dir=save_dir,
                                log_model=False, name=cfg.TAG)

    # Set random seed for reproducibility
    seed = 42
    L.seed_everything(seed, workers=True)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    trainloader, valloader = prepare_dataloaders(cfg)

    if cfg.PRETRAINED.LOAD_WEIGHTS:
        # Load model weights from checkpoint
        l_module = TrainingModule.load_from_checkpoint(os.path.join(cfg.PRETRAINED.PATH,
                                                                    cfg.PRETRAINED.CKPT),
                                                       hparams=hparams,
                                                       cfg=cfg)
        print(f'Loaded model from {cfg.PRETRAINED.PATH}{cfg.PRETRAINED.CKPT}')
    else:
        l_module = TrainingModule(hparams, cfg)


    chkpt_callback = ModelCheckpoint(dirpath=os.path.join(save_dir,'checkpoints'),
                                     monitor='vpq',
                                     save_top_k=3,
                                     mode='max',
                                     filename='model-{epoch}-{vpq:.4f}')

    lr_monitor = LearningRateMonitor(logging_interval='epoch')

    trainer = L.Trainer(
        accelerator=cfg.ACCELERATOR,
        devices=cfg.DEVICES,
        precision=cfg.PRECISION,
        sync_batchnorm=True,
        gradient_clip_val=cfg.GRAD_NORM_CLIP,
        max_epochs=cfg.EPOCHS,
        logger=wdb_logger,
        log_every_n_steps=cfg.LOGGING_INTERVAL,
        callbacks=[chkpt_callback, lr_monitor],
        profiler='simple',
        accumulate_grad_batches=16//cfg.BATCHSIZE
    )
    
    if cfg.PRETRAINED.RESUME_TRAINING:
        trainer.fit(l_module, trainloader, valloader,
                    ckpt_path=os.path.join(cfg.PRETRAINED.PATH,cfg.PRETRAINED.CKPT))
    else:
        trainer.fit(l_module, trainloader, valloader)

    # Free memory
    del l_module
    del trainer
    del trainloader
    del valloader
    torch.cuda.empty_cache()



if __name__ == "__main__":
    # Create parser with one argument
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='b0_short', required=True,
                        choices=['tiny_short', 'b0_short','tiny_long', 'b0_long',
                                 'tiny_short_resnet18',
                                 'tiny_short_timm_mobilenetv4_hybrid_large',
                                 'tiny_short_timm_convnext_tiny'])
    args = parser.parse_args()



    main(args)