import torch
import torch.nn as nn
from torch.optim import SGD, Adam, AdamW
from torch.optim.lr_scheduler import LambdaLR, StepLR

from pytorch_lightning.core import LightningModule
from torchmetrics import ConfusionMatrix, MetricCollection

from examples.MeanAccuracy import MeanAccuracy
from examples.Accuracy import Accuracy
from examples.MeanIoU import MeanIoU

from examples.str2bool import str2bool
from examples.basic_blocks import MLP

from examples.utils import save_pc

class LambdaStepLR(LambdaLR):
  def __init__(self, optimizer, lr_lambda, last_step=-1):
    super(LambdaStepLR, self).__init__(optimizer, lr_lambda, last_step)

  @property
  def last_step(self):
    """Use last_epoch for the step counter"""
    return self.last_epoch

  @last_step.setter
  def last_step(self, v):
    self.last_epoch = v

class SquaredLR(LambdaStepLR):
  """ Used for SGD Lars"""
  def __init__(self, optimizer, max_iter, last_step=-1):
    super(SquaredLR, self).__init__(optimizer, lambda s: (1 - s / (max_iter + 1))**2, last_step)

def to_precision(inputs, precision):
    # print(precision)
    if precision == 'mixed':
        dtype = torch.float16
    elif precision == 32:
        dtype = torch.float32
    elif precision == 64:
        dtype = torch.float64
    for key, value in inputs:
        if isinstance(value, list):
            loutputs = []
            for loutput in value:
                if loutput is not None:
                    loutputs.append(loutput.to(dtype))
                else:
                    loutputs.append(loutput)
            inputs[key] = loutputs
            # print(loutputs)
        else:
            inputs[key] = value.to(dtype)
    return inputs

class BaseSegmentationModule(LightningModule):
    def __init__(self, num_classes, overlap_factor, voxel_size, **kwargs):
        super().__init__()
        self.save_hyperparameters(kwargs)
        for name, value in kwargs.items():
            if name != "self":
                try:
                    setattr(self, name, value)
                except:
                    print(name, value)
        self.num_classes = num_classes
        self.overlap_factor = overlap_factor
        self.voxel_size = voxel_size
        self.criterion = nn.CrossEntropyLoss(weight=self.label_weights, ignore_index=-100)
        metrics = MetricCollection({
                                    'acc': Accuracy(dist_sync_on_step=True),
                                    'macc': MeanAccuracy(num_classes=self.num_classes, dist_sync_on_step=True),
                                    'miou': MeanIoU(num_classes=self.num_classes, dist_sync_on_step=True),
                                    })
        # metrics = MetricCollection({
        #                             'acc': Accuracy(dist_sync_on_step=True),
        #                             'macc': Accuracy(num_classes=self.num_classes, average='macro', dist_sync_on_step=True),
        #                             'miou': IoU(num_classes=self.num_classes, dist_sync_on_step=True),
        #                             })
        self.train_metrics = metrics.clone(prefix='train_')
        self.val_metrics = metrics.clone(prefix='val_')
        conf_metrics = MetricCollection({'ConfusionMatrix': ConfusionMatrix(num_classes=self.num_classes),
                                         'NormalizedConfusionMatrix': ConfusionMatrix(num_classes=self.num_classes, normalize='true')
                                        })
        self.train_conf_metrics = conf_metrics.clone(prefix='train_')
        self.val_conf_metrics = conf_metrics.clone(prefix='val_')

    def training_step(self, batch, batch_idx):
        logits = self(batch)
        target = torch.cat(batch['labels'], dim=0).long()
        train_loss = self.criterion(logits, target)

        self.log('train_loss', train_loss, sync_dist=True, prog_bar=True, on_step=False, on_epoch=True)
        preds = logits.argmax(dim=-1)
        # target = torch.cat(target, dim=0).long()
        valid_targets = target != -100

        if self.global_step % 1 == 0:
            torch.cuda.empty_cache()

        # print("train_metrics")
        # print(preds, target)
        self.train_metrics(preds[valid_targets], target[valid_targets])
        # print("done train_metrics")
        self.log_dict(self.train_metrics, sync_dist=True, prog_bar=False, on_step=False, on_epoch=True)
        self.train_conf_metrics(preds[valid_targets], target[valid_targets])
        self.log_dict(self.train_conf_metrics, sync_dist=True, prog_bar=False, on_step=False, on_epoch=False)

        return train_loss

    # def training_step_end(self, outputs):
    #     #update and log
        

    def validation_step(self, batch, batch_idx):
        logits = self(batch)
        target = torch.cat(batch['labels'], dim=0).long()
        val_loss = self.criterion(logits, target)
        self.log('val_loss', val_loss, sync_dist=True, prog_bar=True, on_step=False, on_epoch=True)
        preds = logits.argmax(dim=-1)
        # 
        valid_targets = target != -100

        if self.global_step % 1 == 0:
            torch.cuda.empty_cache()

        self.val_metrics(preds[valid_targets], target[valid_targets])
        self.log_dict(self.val_metrics, sync_dist=True, prog_bar=True, on_step=False, on_epoch=True)
        self.val_conf_metrics(preds[valid_targets], target[valid_targets])
        self.log_dict(self.val_conf_metrics, sync_dist=True, prog_bar=False, on_step=False, on_epoch=False)

        return val_loss

    # def validation_step_end(self, outputs):
    #     #update and log
    #     # print("val step end")
        

    def configure_optimizers(self):
        if self.optimizer == 'SGD':
            optimizer = SGD(
                    self.parameters(),
                    lr=self.lr,
                    momentum=self.sgd_momentum,
                    dampening=self.sgd_dampening,
                    weight_decay=self.weight_decay)
        elif self.optimizer == 'Adam':
            optimizer = Adam(
                    self.parameters(),
                    lr=self.lr,
                    betas=(self.adam_beta1, self.adam_beta2),
                    weight_decay=self.weight_decay)
        elif self.optimizer == 'AdamW':
            optimizer = AdamW(
                    self.parameters(),
                    lr=self.lr,
                    betas=(self.adam_beta1, self.adam_beta2),
                    weight_decay=self.weight_decay)
        else:
            logging.error('Optimizer type not supported')
            raise ValueError('Optimizer type not supported')


        if self.scheduler == 'StepLR':
            scheduler = StepLR(
                optimizer, step_size=self.step_size, gamma=self.step_gamma, last_epoch=-1)
        elif self.scheduler == 'SquaredLR':
            scheduler = SquaredLR(optimizer, max_iter=self.max_iter, last_step=-1)
        elif self.scheduler == 'None':
            return [optimizer]
        else:
            logging.error('Scheduler not supported')
            raise ValueError('Scheduler not supported')

        lr_sched = {
                    'scheduler': scheduler,
                    'interval': 'step'  # called after each training step
                    }

        return [optimizer], [lr_sched]

    def remove_conf_matrices(self, metrics_dict):
        new_metrics_dict = dict(metrics_dict)
        conf_matrices = []
        for key in new_metrics_dict.keys():
            if "ConfusionMatrix" in key:
                conf_matrices.append(key)
        for key in conf_matrices:
            new_metrics_dict.pop(key)
        return new_metrics_dict

    #def convert_sync_batchnorm(self):
    #    self.model = ME.MinkowskiSyncBatchNorm.convert_sync_batchnorm(self.model)

    @staticmethod
    def add_argparse_args(parent_parser):
        parser = parent_parser.add_argument_group("BaseSegModel")
        parser.add_argument("--save_pcs", type=str2bool, nargs='?', const=True, default=False)

        # Optimizer arguments
        parser.add_argument('--optimizer', type=str, default='SGD', choices=['SGD', 'Adam', 'AdamW'])
        parser.add_argument('--lr', type=float, default=1e-1)
        parser.add_argument('--sgd_momentum', type=float, default=0.9)
        parser.add_argument('--sgd_dampening', type=float, default=0.1)
        parser.add_argument('--adam_beta1', type=float, default=0.9)
        parser.add_argument('--adam_beta2', type=float, default=0.999)
        parser.add_argument('--weight_decay', type=float, default=1e-4)
        parser.add_argument('--iter_size', type=int, default=1, help='accumulate gradient')

        # Scheduler
        parser.add_argument('--scheduler', type=str, default='SquaredLR')
        parser.add_argument('--max_iter', type=int, default=6e4)
        parser.add_argument('--step_size', type=int, default=2e4)
        parser.add_argument('--step_gamma', type=float, default=0.1)
        parser.add_argument('--poly_power', type=float, default=0.9)
        parser.add_argument('--exp_gamma', type=float, default=0.95)
        parser.add_argument('--exp_step_size', type=float, default=445)
        return parent_parser
