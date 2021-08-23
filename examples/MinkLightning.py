import torch
import torch.nn as nn
from torch.optim import SGD, Adam
from torch.optim.lr_scheduler import LambdaLR, StepLR

from pytorch_lightning.core import LightningModule
import MinkowskiEngine as ME
from examples.minkunet import MinkUNet34C
from examples.MeanAccuracy import MeanAccuracy
from examples.MeanIoU import MeanIoU
from examples.sam import SAM
from examples.str2bool import str2bool

from pytorch_lightning.metrics import Accuracy, ConfusionMatrix, MetricCollection


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
    if precision == 16:
        dtype = torch.float16
    elif precision == 32:
        dtype = torch.float32
    elif precision == 64:
        dtype = torch.float64
    return tuple([input.to(dtype) for input in inputs])

class MinkowskiSegmentationModule(LightningModule):
    def __init__(self, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        for name, value in kwargs.items():
            if name != "self":
                try:
                    setattr(self, name, value)
                except:
                    print(name, value)
        self.criterion = nn.CrossEntropyLoss(ignore_index=-100)
        self.model = MinkUNet34C(self.in_channels, self.out_channels)
        metrics = MetricCollection({'acc': Accuracy(),
                                    'macc': MeanAccuracy(num_classes=self.out_channels),
                                    'miou': MeanIoU(num_classes=self.out_channels)})
        self.train_metrics = metrics.clone(prefix='train_')
        self.val_metrics = metrics.clone(prefix='val_')
        conf_metrics = MetricCollection({'ConfusionMatrix': ConfusionMatrix(num_classes=self.out_channels),
                                         'NormalizedConfusionMatrix': ConfusionMatrix(num_classes=self.out_channels, normalize='true')
                                        })
        self.train_conf_metrics = conf_metrics.clone(prefix='train_')
        self.val_conf_metrics = conf_metrics.clone(prefix='val_')

        self.automatic_optimization = not self.use_sam

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        # print("Train ", batch)
        coords, feats, target = batch['coords'], batch['feats'], batch['labels']
        coords, feats = to_precision((coords, feats), self.trainer.precision)     
        in_field = ME.TensorField(
            features=feats,
            coordinates=coords,
            quantization_mode=ME.SparseTensorQuantizationMode.UNWEIGHTED_AVERAGE,
            minkowski_algorithm=ME.MinkowskiAlgorithm.SPEED_OPTIMIZED,
            device=self.device,
        )
        sinput = in_field.sparse()
        
        # if self.global_step % 10 == 0:
        #     torch.cuda.empty_cache()

        logits = self(sinput).slice(in_field).F
        train_loss = self.criterion(logits, target)

        if self.use_sam:
            optimizer = self.optimizers()
            # first backward pass
            self.manual_backward(train_loss, optimizer)
            optimizer.first_step(zero_grad=True)

            self.disable_bn()
            # second forward-backward pass
            logits2 = self(sinput).slice(in_field).F
            loss2 = self.criterion(logits2, target)
            self.manual_backward(loss_2, optimizer)
            optimizer.second_step(zero_grad=True)
            self.enable_bn()

        self.log('train_loss', train_loss, sync_dist=True, prog_bar=True, on_step=True, on_epoch=False)
        preds = logits.argmax(dim=-1)
        valid_targets = target != -100
        return {'loss': train_loss, 'preds': preds[valid_targets], 'target': target[valid_targets]}

    def training_step_end(self, outputs):
        #update and log
        self.train_metrics(outputs['preds'], outputs['target'])
        self.log_dict(self.train_metrics, prog_bar=False, on_step=False, on_epoch=True)
        self.train_conf_metrics(outputs['preds'], outputs['target'])
        self.log_dict(self.train_conf_metrics, prog_bar=False, on_step=False, on_epoch=False)

    def validation_step(self, batch, batch_idx):
        # print("Val ", batch)
        coords, feats, target = batch['coords'], batch['feats'], batch['labels']
        coords, feats = to_precision((coords, feats), self.trainer.precision)
        # print(target.min(), target.max(), target)
        in_field = ME.TensorField(
            features=feats,
            coordinates=coords,
            quantization_mode=ME.SparseTensorQuantizationMode.UNWEIGHTED_AVERAGE,
            minkowski_algorithm=ME.MinkowskiAlgorithm.SPEED_OPTIMIZED,
            device=self.device,
        )
        sinput = in_field.sparse()
        logits = self(sinput).slice(in_field).F
        val_loss = self.criterion(logits, target)
        self.log('val_loss', val_loss, sync_dist=True, prog_bar=True, on_step=True, on_epoch=False)
        preds = logits.argmax(dim=-1)
        valid_targets = target != -100
        return {'loss': val_loss, 'preds': preds[valid_targets], 'target': target[valid_targets]}

    def validation_step_end(self, outputs):
        #update and log
        self.val_metrics(outputs['preds'], outputs['target'])
        self.log_dict(self.val_metrics, prog_bar=True, on_step=False, on_epoch=True)
        self.val_conf_metrics(outputs['preds'], outputs['target'])
        self.log_dict(self.val_conf_metrics, prog_bar=False, on_step=False, on_epoch=False)

    def configure_optimizers(self):
        if self.optimizer == 'SGD':
            if self.use_sam:
                optimizer = SAM(self.parameters(),
                                SGD, lr=self.lr,
                                adaptive=self.sam_adaptive,
                                rho=self.sam_rho,
                                momentum=self.sgd_momentum,
                                dampening=self.sgd_dampening,
                                weight_decay=self.weight_decay)
            else:
                optimizer = SGD(
                    self.parameters(),
                    lr=self.lr,
                    momentum=self.sgd_momentum,
                    dampening=self.sgd_dampening,
                    weight_decay=self.weight_decay)
        elif self.optimizer == 'Adam':
            if self.use_sam:
                optimizer = SAM(self.parameters(),
                                Adam, lr=self.lr,
                                adaptive=self.sam_adaptive,
                                rho=self.sam_rho,
                                betas=(self.adam_beta1, self.adam_beta2),
                                weight_decay=self.weight_decay)
            else:
                optimizer = Adam(
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

    def disable_bn(self):
        for module in self.modules():
            if isinstance(module, nn.BatchNorm1d) or isinstance(module, ME.MinkowskiBatchNorm):
                module.eval()

    def enable_bn(self):
        self.train()

    @staticmethod
    def add_argparse_args(parent_parser):
        parser = parent_parser.add_argument_group("MinkSegModel")
        parser.add_argument("--in_channels", type=int, default=3)
        parser.add_argument("--out_channels", type=int, default=20)

        # Optimizer arguments
        parser.add_argument('--optimizer', type=str, default='SGD', choices=['SGD', 'Adam'])
        parser.add_argument('--lr', type=float, default=1e-1)
        parser.add_argument('--sgd_momentum', type=float, default=0.9)
        parser.add_argument('--sgd_dampening', type=float, default=0.1)
        parser.add_argument('--adam_beta1', type=float, default=0.9)
        parser.add_argument('--adam_beta2', type=float, default=0.999)
        parser.add_argument('--weight_decay', type=float, default=1e-4)
        parser.add_argument('--param_histogram_freq', type=int, default=100)
        # parser.add_argument('--save_param_histogram', type=str2bool, default=False)
        parser.add_argument('--iter_size', type=int, default=1, help='accumulate gradient')
        parser.add_argument('--bn_momentum', type=float, default=0.02)
        parser.add_argument("--use_sam", type=str2bool, nargs='?', const=True, default=False)
        parser.add_argument("--sam_adaptive", type=str2bool, nargs='?', const=True, default=False)
        parser.add_argument('--sam_rho', type=float, default=0.05)

        # Scheduler
        parser.add_argument('--scheduler', type=str, default='SquaredLR')
        parser.add_argument('--max_iter', type=int, default=6e4)
        parser.add_argument('--step_size', type=int, default=2e4)
        parser.add_argument('--step_gamma', type=float, default=0.1)
        parser.add_argument('--poly_power', type=float, default=0.9)
        parser.add_argument('--exp_gamma', type=float, default=0.95)
        parser.add_argument('--exp_step_size', type=float, default=445)
        return parent_parser


