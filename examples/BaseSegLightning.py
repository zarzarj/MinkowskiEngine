import torch
import torch.nn as nn
from torch.optim import SGD, Adam, AdamW
from torch.optim.lr_scheduler import LambdaLR, StepLR, OneCycleLR

from pytorch_lightning.core import LightningModule
from torchmetrics import ConfusionMatrix, MetricCollection, IoU, Accuracy as Acc

from examples.MeanAccuracy import MeanAccuracy
from examples.Accuracy import Accuracy
from examples.MeanIoU import MeanIoU

from examples.str2bool import str2bool
from examples.basic_blocks import MLP

from examples.utils import save_pc
import gc
import traceback

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

class BaseSegmentationModule(LightningModule):
    def __init__(self, **kwargs):
        super().__init__()
        self.save_hyperparameters(kwargs)
        for name, value in kwargs.items():
            if name != "self":
                try:
                    setattr(self, name, value)
                except:
                    print(name, value)
        self.criterion = nn.CrossEntropyLoss(weight=self.label_weights, ignore_index=-1)
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
        self.train_class_iou = IoU(num_classes=self.num_classes, reduction='none', dist_sync_on_step=True)
        self.val_class_iou = IoU(num_classes=self.num_classes, reduction='none', dist_sync_on_step=True)
        self.train_class_acc = Acc(num_classes=self.num_classes, average='none', dist_sync_on_step=True)
        self.val_class_acc = Acc(num_classes=self.num_classes, average='none', dist_sync_on_step=True)

    def training_step(self, batch, batch_idx):
        logits = self(batch)
        # print(logits.shape)
        target = batch['labels'].long()
        valid_targets = target != -1
        target = target[valid_targets]
        if len(logits.shape) == 3:
            logits = logits[:, valid_targets]
            preds = logits.detach().argmax(dim=-1)[0]
        else:
            logits = logits[valid_targets]
            preds = logits.detach().argmax(dim=-1)

        train_loss = self.criterion(logits, target)
        if type(train_loss) is dict:
            self.log('train_loss', train_loss['loss'], sync_dist=True, prog_bar=True, on_step=False, on_epoch=True)
        else:
            self.log('train_loss', train_loss, sync_dist=True, prog_bar=True, on_step=False, on_epoch=True)
        
        self.train_metrics(preds, target)
        self.log_dict(self.train_metrics, sync_dist=True, prog_bar=False, on_step=False, on_epoch=True)
        self.train_class_iou(preds, target)
        self.train_class_acc(preds, target)
        # self.log_dict(dict(zip(['train_'+l+'_iou' for l in self.trainer.datamodule.class_labels], self.train_class_iou.compute())), sync_dist=True, prog_bar=False, on_step=False, on_epoch=True)
        # self.train_conf_metrics(preds, target)
        # self.log_dict(self.train_conf_metrics, sync_dist=True, prog_bar=False, on_step=False, on_epoch=False)
        return train_loss
        
    def validation_step(self, batch, batch_idx):
        logits = self(batch)
        # print(logits.shape)

        target = batch['labels'].long()
        valid_targets = target != -1

        target = target[valid_targets]
        # print(logits.shape)
        if len(logits.shape) == 3:
            logits = logits[:, valid_targets]
            preds = logits.detach().argmax(dim=-1)[0]
        else:
            logits = logits[valid_targets]
            preds = logits.detach().argmax(dim=-1)

        val_loss = self.criterion(logits, target)
        if type(val_loss) is dict:
            self.log('val_loss', val_loss['loss'], sync_dist=True, prog_bar=True, on_step=False, on_epoch=True)
        else:
            self.log('val_loss', val_loss, sync_dist=True, prog_bar=True, on_step=False, on_epoch=True)

        self.val_metrics(preds, target)
        self.log_dict(self.val_metrics, sync_dist=True, prog_bar=True, on_step=False, on_epoch=True)
        self.val_class_iou(preds, target)
        self.val_class_acc(preds, target)
        # self.log_dict(dict(zip(['val_'+ l+'_iou' for l in self.trainer.datamodule.class_labels], self.val_class_iou.compute())), sync_dist=True, prog_bar=False, on_step=False, on_epoch=True)
        # self.val_conf_metrics(preds, target)
        # self.log_dict(self.val_conf_metrics, sync_dist=True, prog_bar=False, on_step=False, on_epoch=False)
        return val_loss
        
    def configure_optimizers(self):
        if self.split_wd:
            decay, no_decay = [], []
            for name, param in self.named_parameters():
                # print(name)
                if not param.requires_grad: continue # frozen weights                 
                if len(param.shape) == 1 or name.endswith(".bias") or ".bn" in name: no_decay.append(param)
                else: decay.append(param)
            params = [{'params': no_decay, 'weight_decay': 0.}, {'params': decay, 'weight_decay': self.weight_decay}]
        else:
            params = self.parameters()
        if self.optimizer == 'SGD':
            optimizer = SGD(
                    params,
                    lr=self.lr,
                    momentum=self.sgd_momentum,
                    dampening=self.sgd_dampening,
                    weight_decay=self.weight_decay)
        elif self.optimizer == 'Adam':
            optimizer = Adam(
                    params,
                    lr=self.lr,
                    betas=(self.adam_beta1, self.adam_beta2),
                    weight_decay=self.weight_decay)
        elif self.optimizer == 'AdamW':
            optimizer = AdamW(
                    params,
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
        elif self.scheduler == 'OneCycleLR':
            scheduler = OneCycleLR(optimizer, max_lr=self.lr, total_steps=self.max_iter)
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
        parser.add_argument('--max_iter', type=int, default=60000)
        parser.add_argument('--step_size', type=int, default=2e4)
        parser.add_argument('--step_gamma', type=float, default=0.1)
        parser.add_argument('--poly_power', type=float, default=0.9)
        parser.add_argument('--exp_gamma', type=float, default=0.95)
        parser.add_argument('--exp_step_size', type=float, default=445)

        parser.add_argument("--split_wd", type=str2bool, nargs='?', const=True, default=False)
        return parent_parser
