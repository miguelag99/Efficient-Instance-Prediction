import torch

from prediction.data.nuscenes_dataset import NuscenesDataset


def prepare_dataloaders(cfg, return_dataset=False, return_orig_images=False):
    """
    Prepare the NuScenes dataloader
    """
    version = cfg.DATASET.VERSION

    # TODO: pass nuscenes object to each loader instead of creating it inside
    train_data = NuscenesDataset(cfg, mode = 'train',
                                 return_orig_images=return_orig_images)
    val_data = NuscenesDataset(cfg, mode = 'val',
                               return_orig_images=return_orig_images)

    if cfg.DATASET.VERSION == 'mini':
        train_data.indices = train_data.indices[:10]
        val_data.indices = val_data.indices[:10]
     
    nworkers = cfg.N_WORKERS
    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=cfg.BATCHSIZE, shuffle=True,
        num_workers=nworkers, pin_memory=True, drop_last=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_data, batch_size=cfg.BATCHSIZE, shuffle=False,
        num_workers=nworkers, pin_memory=True, drop_last=True)

    if return_dataset:
        return train_loader, val_loader, train_data, val_data
    else:
        return train_loader, val_loader