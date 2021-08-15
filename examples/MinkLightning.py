import torch
import torch.nn as nn
from torch.optim import SGD

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
        self.criterion = nn.CrossEntropyLoss()
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
        return SGD(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)

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
        parser.add_argument("--optimizer_name", type=str, default='SGD')
        parser.add_argument("--lr", type=float, default=1e-3)
        parser.add_argument("--weight_decay", type=float, default=1e-5)
        # parser.add_argument("--voxel_size", type=float, default=0.02)
        return parent_parser