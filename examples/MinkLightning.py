import torch
import torch.nn as nn

import MinkowskiEngine as ME
from examples.minkunet import MinkUNet34C
from examples.str2bool import str2bool
from examples.basic_blocks import MLP
from examples.BaseSegLightning import BaseSegmentationModule

class MinkowskiSegmentationModule(BaseSegmentationModule):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model = MinkUNet34C(self.in_channels, self.out_channels)
        if self.use_seg_head:
            self.seg_head = nn.Sequential(MLP(self.mlp_channels, dropout=self.seg_head_dropout),
                                      nn.Conv1d(self.mlp_channels[-1], self.out_channels, kernel_size=1, bias=True)
                                      # nn.Linear(self.mlp_channels[-1], self.out_channels)
                                      )

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

        logits = self(sinput).slice(in_field).F
        train_loss = self.criterion(logits, target)

        self.log('train_loss', train_loss, sync_dist=True, prog_bar=True, on_step=False, on_epoch=True)
        preds = logits.argmax(dim=-1)
        valid_targets = target != -100

        if self.global_step % 10 == 0:
            torch.cuda.empty_cache()

        return {'loss': train_loss, 'preds': preds[valid_targets], 'target': target[valid_targets]}
    
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
        self.log('val_loss', val_loss, sync_dist=True, prog_bar=True, on_step=False, on_epoch=True)
        preds = logits.argmax(dim=-1)
        valid_targets = target != -100
        return {'loss': val_loss, 'preds': preds[valid_targets], 'target': target[valid_targets]}

    @staticmethod
    def add_argparse_args(parent_parser):
        parent_parser = BaseSegmentationModule.add_argparse_args(parent_parser)
        parser = parent_parser.add_argument_group("MinkSegModel")
        parser.add_argument("--use_seg_head", type=str2bool, nargs='?', const=True, default=False)
        return parent_parser


