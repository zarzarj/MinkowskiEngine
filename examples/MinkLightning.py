import torch
import torch.nn as nn
from torch.optim import SGD, Adam
from torch.optim.lr_scheduler import LambdaLR, StepLR

from pytorch_lightning.core import LightningModule
import MinkowskiEngine as ME
from examples.minkunet import MinkUNet34C
from examples.MeanAccuracy import MeanAccuracy
from examples.MeanIoU import MeanIoU
from pytorch_lightning.metrics import Accuracy, ConfusionMatrix, MetricCollection


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
                                    'ConfusionMatrix': ConfusionMatrix(num_classes=self.out_channels),
                                    'NormalizedConfusionMatrix': ConfusionMatrix(num_classes=self.out_channels, normalize='true'),
                                    'macc': MeanAccuracy(num_classes=self.out_channels),
                                    'miou': MeanIoU(num_classes=self.out_channels)})
        self.train_metrics = metrics.clone(prefix='train_')
        self.val_metrics = metrics.clone(prefix='val_')

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        coords, feats, target = batch['coords'], batch['feats'], batch['labels']
        coords, feats, target = to_precision((coords, feats, target), self.trainer.precision)
        target = target.long()
        in_field = ME.TensorField(
            features=feats,
            coordinates=coords,
            quantization_mode=ME.SparseTensorQuantizationMode.UNWEIGHTED_AVERAGE,
            minkowski_algorithm=ME.MinkowskiAlgorithm.SPEED_OPTIMIZED,
            device=self.device,
        )
        sinput = in_field.sparse()
        if self.global_step % 10 == 0:
            torch.cuda.empty_cache()
        logits = self(sinput).slice(in_field).F
        print(logits.shape)
        train_loss = self.criterion(logits, target)
        self.log('train_loss', train_loss, sync_dist=True, prog_bar=True, on_step=True, on_epoch=True)
        self.log_metrics(logits, target, self.train_metrics)
        return train_loss

    def on_train_epoch_end(self, unused=None):
        metrics_dict = self.train_metrics.compute()
        metrics_dict = self.remove_conf_matrices(metrics_dict)
        self.log_dict(metrics_dict, sync_dist=True, prog_bar=False)
        self.train_metrics.reset()

    def validation_step(self, batch, batch_idx):
        coords, feats, target = batch['coords'], batch['feats'], batch['labels']
        coords, feats, target = to_precision((coords, feats, target), self.trainer.precision)
        target = target.long()
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
        self.log('val_loss', val_loss, sync_dist=True, prog_bar=True, on_step=True, on_epoch=True)
        self.log_metrics(logits, target, self.val_metrics)
        return val_loss

    def on_validation_epoch_end(self, unused=None):
        metrics_dict = self.val_metrics.compute()
        metrics_dict = self.remove_conf_matrices(metrics_dict)
        self.log_dict(metrics_dict, sync_dist=True, prog_bar=True)
        self.val_metrics.reset()

    def configure_optimizers(self):
        if self.optimizer == 'SGD':
            optimizer = SGD(
                self.model.parameters(),
                lr=self.lr,
                momentum=self.sgd_momentum,
                dampening=self.sgd_dampening,
                weight_decay=self.weight_decay)
        elif self.optimizer == 'Adam':
            optimizer = Adam(
                self.model.parameters(),
                lr=self.lr,
                betas=(self.adam_beta1, self.adam_beta2),
                weight_decay=self.weight_decay)
        else:
            logging.error('Optimizer type not supported')
            raise ValueError('Optimizer type not supported')

        if self.scheduler == 'StepLR':
            scheduler = StepLR(
                optimizer, step_size=self.step_size, gamma=self.step_gamma, last_epoch=-1)
        else:
            logging.error('Scheduler not supported')

        return [optimizer], [scheduler]

    def log_metrics(self, logits, target, metrics_fn):
        valid_idx = target != -100
        preds = logits.argmax(dim=-1)
        metrics_fn.update(preds[valid_idx], target[valid_idx])

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
        parser = parent_parser.add_argument_group("MinkSegModel")
        parser.add_argument("--in_channels", type=int, default=3)
        parser.add_argument("--out_channels", type=int, default=20)

        # Optimizer arguments
        parser.add_argument('--optimizer', type=str, default='SGD')
        parser.add_argument('--lr', type=float, default=1e-2)
        parser.add_argument('--sgd_momentum', type=float, default=0.9)
        parser.add_argument('--sgd_dampening', type=float, default=0.1)
        parser.add_argument('--adam_beta1', type=float, default=0.9)
        parser.add_argument('--adam_beta2', type=float, default=0.999)
        parser.add_argument('--weight_decay', type=float, default=1e-4)
        parser.add_argument('--param_histogram_freq', type=int, default=100)
        # parser.add_argument('--save_param_histogram', type=str2bool, default=False)
        parser.add_argument('--iter_size', type=int, default=1, help='accumulate gradient')
        parser.add_argument('--bn_momentum', type=float, default=0.02)

        # Scheduler
        parser.add_argument('--scheduler', type=str, default='StepLR')
        parser.add_argument('--max_iter', type=int, default=6e4)
        parser.add_argument('--step_size', type=int, default=2e4)
        parser.add_argument('--step_gamma', type=float, default=0.1)
        parser.add_argument('--poly_power', type=float, default=0.9)
        parser.add_argument('--exp_gamma', type=float, default=0.95)
        parser.add_argument('--exp_step_size', type=float, default=445)
        return parent_parser