import time
import lightning as L
import torch
import torch.nn as nn
import numpy as np
import wandb

from torch.optim.lr_scheduler import ReduceLROnPlateau, PolynomialLR

from prediction.losses import SegmentationLoss, SpatialRegressionLoss
from prediction.metrics import IntersectionOverUnion, PanopticMetric
from prediction.utils.instance import predict_instance_segmentation
from prediction.model.predictor import FullSegformerCustomHead



class TrainingModule(L.LightningModule):
    def __init__(self, hparams, cfg):
        super().__init__()

        self.params = hparams
        self.save_hyperparameters()
        self.cfg = cfg
        self.bs = self.cfg.BATCHSIZE

        self.n_classes = len(self.cfg.SEMANTIC_SEG.WEIGHTS)


        self.model = FullSegformerCustomHead(self.cfg)


        # Bird's-eye view extent in meters
        assert self.cfg.LIFT.X_BOUND[1] > 0 and self.cfg.LIFT.Y_BOUND[1] > 0
        self.spatial_extent = (self.cfg.LIFT.X_BOUND[1], self.cfg.LIFT.Y_BOUND[1])
 

        # Losses
        seg_loss = SegmentationLoss(
            class_weights=torch.Tensor(self.cfg.SEMANTIC_SEG.WEIGHTS),
            use_top_k=self.cfg.SEMANTIC_SEG.USE_TOP_K,
            top_k_ratio=self.cfg.SEMANTIC_SEG.TOP_K_RATIO,
            future_discount=self.cfg.FUTURE_DISCOUNT,
        )
        flow_loss = SpatialRegressionLoss(
            norm=1.5, 
            future_discount=self.cfg.FUTURE_DISCOUNT, 
            ignore_index=self.cfg.DATASET.IGNORE_INDEX,
        )
        self.losses_fn = nn.ModuleDict({
            'segmentation': seg_loss,
            'instance_flow': flow_loss,
        })

        # Uncertainty weighting
        self.model.segmentation_weight = nn.Parameter(torch.tensor(0.0),
                                                      requires_grad=True)
        self.model.flow_weight = nn.Parameter(torch.tensor(0.0), requires_grad=True)

        # Metrics
        self.metric_iou_val = IntersectionOverUnion(n_classes=self.n_classes)
        self.metric_panoptic_val = PanopticMetric(n_classes=self.n_classes)

        self.training_step_count = 0

        # Run time
        self.perception_time, self.prediction_time, self.postprocessing_time = [], [], []

    def shared_step(self, batch, is_train):
            image = batch['image']
            intrinsics = batch['intrinsics']
            extrinsics = batch['extrinsics']
            future_egomotion = batch['future_egomotion']

            # Warp labels
            labels, future_distribution_inputs = self.prepare_future_labels(batch)

            # Forward pass
            output = self.model(image, intrinsics, extrinsics, future_egomotion,
                                future_distribution_inputs)

            # Calculate loss
            loss = self.calculate_loss(output, labels)

            if not is_train:
                # Perform warping-based pixel-level association
                start_time = time.time()
                pred_consistent_instance_seg = predict_instance_segmentation(output, spatial_extent=self.spatial_extent)
                end_time = time.time()

                # Calculate metrics
                self.metric_iou_val(torch.argmax(output['segmentation'].detach(),
                                                 dim=2, keepdims=True)[:, 1:],
                                    labels['segmentation'][:, 1:])
                self.metric_panoptic_val(pred_consistent_instance_seg[:, 1:],
                                         labels['instance'][:, 1:])
            
                # Record run time
                self.perception_time.append(output['perception_time'])
                self.prediction_time.append(output['prediction_time'])
                self.postprocessing_time.append(end_time-start_time)

            return output, labels, loss

    def calculate_loss(self, output, labels):
        loss = {}
        segmentation_factor = 100 / torch.exp(self.model.segmentation_weight)
        loss['segmentation'] = segmentation_factor * self.losses_fn['segmentation'](
            output['segmentation'], 
            labels['segmentation'], 
        )
        loss[f'segmentation_uncertainty'] = 0.5 * self.model.segmentation_weight

        flow_factor = 0.1 / (2*torch.exp(self.model.flow_weight))

        loss['instance_flow'] = flow_factor * self.losses_fn['instance_flow'](
            output['instance_flow'], 
            labels['flow']
        )
        loss['flow_uncertainty'] = 0.5 * self.model.flow_weight

        return loss

    def prepare_future_labels(self, batch):
        labels = {}
        future_distribution_inputs = []

        segmentation_labels = batch['segmentation']
        instance_center_labels = batch['centerness']
        instance_offset_labels = batch['offset']
        instance_flow_labels = batch['flow']
        gt_instance = batch['instance']

        label_time_range = self.model.receptive_field - 2  # See section 3.4 in paper for details.

        segmentation_labels = segmentation_labels[:, label_time_range:].long().contiguous()
        labels['segmentation'] = segmentation_labels
        future_distribution_inputs.append(segmentation_labels)

        gt_instance = gt_instance[:, label_time_range:].long().contiguous()
        labels['instance'] = gt_instance

        instance_center_labels = instance_center_labels[:, label_time_range:].contiguous()
        labels['centerness'] = instance_center_labels
        future_distribution_inputs.append(instance_center_labels)

        instance_offset_labels = instance_offset_labels[:, label_time_range:].contiguous()
        labels['offset'] = instance_offset_labels
        future_distribution_inputs.append(instance_offset_labels)

        instance_flow_labels = instance_flow_labels[:, label_time_range:]
        labels['flow'] = instance_flow_labels
        future_distribution_inputs.append(instance_flow_labels)

        if len(future_distribution_inputs) > 0:
            future_distribution_inputs = torch.cat(future_distribution_inputs, dim=2)
        
        labels['future_egomotion'] = batch['future_egomotion']

        return labels, future_distribution_inputs

    def training_step(self, batch, batch_idx):
        output, labels, loss = self.shared_step(batch, True)
        self.training_step_count += 1
        for key, value in loss.items():
            self.log('train_loss/' + key, value, batch_size = self.bs, sync_dist=True)

        return sum(loss.values())
    
    def validation_step(self, batch, batch_idx):
        output, labels, loss = self.shared_step(batch, False)
        for key, value in loss.items():
            self.log('val_loss/' + key, value, batch_size = self.bs, sync_dist=True)

    def test_step(self, batch, batch_idx):
        output, labels, loss = self.shared_step(batch, False)
        for key, value in loss.items():
            self.log('test_loss/' + key, value, batch_size = self.bs, sync_dist=True)

    def shared_epoch_end(self, is_train):
            # Log per class iou metrics
            class_names = ['background', 'dynamic']
            if not is_train:
                print("========================== Metrics ==========================")
                scores = self.metric_iou_val.compute()
                
                for key, value in zip(class_names, scores):
                    self.log('metrics/val_iou_' + key, value,
                             batch_size = self.bs, sync_dist=True)
                    print(f"val_iou_{key}: {value}")
                self.metric_iou_val.reset()

                scores = self.metric_panoptic_val.compute()

                for key, value in scores.items():
                    for instance_name, score in zip(class_names, value):
                        if instance_name != 'background':
                            self.log(f'metrics/val_{key}_{instance_name}', score.item(),
                                     batch_size = self.bs, sync_dist=True)
                            print(f"val_{key}_{instance_name}: {score.item()}")
                        # Log VPQ metric for the model checkpoint monitor 
                        if key == 'pq' and instance_name == 'dynamic':
                            self.log('vpq', score.item(), batch_size = self.bs,
                                     sync_dist=True)
                self.metric_panoptic_val.reset()

                print("========================== Runtime ==========================")
                perception_time = sum(self.perception_time) / (len(self.perception_time) + 1e-8)
                prediction_time = sum(self.prediction_time) / (len(self.prediction_time) + 1e-8)
                postprocessing_time = sum(self.postprocessing_time) / (len(self.postprocessing_time) + 1e-8)
                print(f"perception_time: {perception_time}")
                print(f"prediction_time: {prediction_time}")
                print(f"postprocessing_time: {postprocessing_time}")
                print(f"total_time: {perception_time + prediction_time + postprocessing_time}")
                print("=============================================================")
                self.perception_time, self.prediction_time, self.postprocessing_time = [], [], []

            self.log('weights/segmentation_weight', 1 / (torch.exp(self.model.segmentation_weight)), sync_dist=True)
            self.log('weights/flow_weight', 1 / (2 * torch.exp(self.model.flow_weight)), sync_dist=True)

    def on_train_epoch_end(self):
        self.shared_epoch_end(True)
    
    def on_validation_epoch_end(self):
        self.shared_epoch_end(False)
    
    def on_test_epoch_end(self):
        self.shared_epoch_end(False)

    def configure_optimizers(self):
        params = self.model.parameters()
        optim_name = self.cfg.OPTIMIZER.TYPE
        sched_name = self.cfg.SCHEDULER.TYPE
        if optim_name == 'AdamW':
            optimizer = torch.optim.AdamW(
                params,lr=self.cfg.OPTIMIZER.LR,
                weight_decay=self.cfg.OPTIMIZER.WEIGHT_DECAY
            )
        elif optim_name == 'Adam':
            optimizer = torch.optim.Adam(
                params, lr=self.cfg.OPTIMIZER.LR,
                weight_decay=self.cfg.OPTIMIZER.WEIGHT_DECAY
            )
        elif optim_name == 'AdamWScheduleFree':
            import schedulefree
            optimizer = schedulefree.AdamWScheduleFree(
                params, lr=self.cfg.OPTIMIZER.LR,
                weight_decay=self.cfg.OPTIMIZER.WEIGHT_DECAY
            )
            return optimizer
        else:
            raise NotImplementedError

        if sched_name == 'ReduceLROnPlateau':
            scheduler = {
                'scheduler': ReduceLROnPlateau(optimizer,
                                               mode='min', factor=0.1, patience=2),
            }
        elif sched_name == 'PolynomialLR':
            scheduler = {
                'scheduler': PolynomialLR(optimizer, total_iters=self.cfg.EPOCHS+1,
                                          power=1, last_epoch=-1),
            }
        elif sched_name == '':
            return optimizer
        
        else:
            raise NotImplementedError

        scheduler['monitor'] = 'val_loss/segmentation'
        scheduler['interval'] = 'epoch'
        scheduler['frequency'] = 1

        return [optimizer], [scheduler]


