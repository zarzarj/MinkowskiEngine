import torch
import torch.nn as nn

import MinkowskiEngine as ME
from examples.minkunet import MinkUNet34C
from examples.str2bool import str2bool
from examples.basic_blocks import MLP
from examples.BaseSegLightning import BaseSegmentationModule

import numpy as np

def to_precision(inputs, precision):
    if precision == 16:
        dtype = torch.float16
    elif precision == 32:
        dtype = torch.float32
    elif precision == 64:
        dtype = torch.float64
    return tuple([input.to(dtype) for input in inputs])

class MinkowskiSegmentationModule(BaseSegmentationModule):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if self.use_seg_head:
            self.mlp_channels = [int(i) for i in self.mlp_channels.split(',')]
            if self.relative_mlp_channels:
                self.mlp_channels = (self.mlp_extra_in_channels) * np.array(self.mlp_channels)
                # print(self.mlp_channels)
            else:
                self.mlp_channels = [self.mlp_extra_in_channels] + self.mlp_channels

            self.model = MinkUNet34C(self.in_channels, self.mlp_extra_in_channels)
            seg_head_list = []
            if self.seg_head_in_bn:
                seg_head_list.append(norm_layer(norm_type='batch', nc=self.mlp_channels[0]))
            seg_head_list += [MLP(self.mlp_channels, dropout=self.seg_head_dropout),
                             nn.Conv1d(self.mlp_channels[-1], self.out_channels, kernel_size=1, bias=True)]
            self.seg_head = nn.Sequential(*seg_head_list)
        else:
            self.model = MinkUNet34C(self.in_channels, self.out_channels)

        if self.pretrained_minkunet_ckpt is not None:
            print("Loading checkpoint from ", self.pretrained_minkunet_ckpt)
            pretrained_ckpt = torch.load(self.pretrained_minkunet_ckpt)
            if 'state_dict' in pretrained_ckpt:
                pretrained_ckpt = pretrained_ckpt['state_dict']
            if self.in_channels != 3:
                del pretrained_ckpt['conv0p1s1.kernel']
            if self.mlp_extra_in_channels != 20:
                del pretrained_ckpt['final.kernel']
                del pretrained_ckpt['final.bias']
            self.model.load_state_dict(pretrained_ckpt, strict=False)

    def forward(self, x):
        # for name, val in self.model.named_parameters():
            # print(name, val)
            # break
        out = self.model(x)
        if self.use_seg_head:
            # print(out._F.shape)
            out._F = self.seg_head(out._F.unsqueeze(-1)).squeeze(-1)
        return out

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
        # print(feats.shape)
        coords, feats = to_precision((coords, feats), self.trainer.precision)

        # print(coords, feats)
        # print(target.min(), target.max(), target)
        in_field = ME.TensorField(
            features=feats,
            coordinates=coords,
            quantization_mode=ME.SparseTensorQuantizationMode.UNWEIGHTED_AVERAGE,
            minkowski_algorithm=ME.MinkowskiAlgorithm.SPEED_OPTIMIZED,
            device=self.device,
        )
        sinput = in_field.sparse()
        logits = self.model(sinput).slice(in_field).F
        val_loss = self.criterion(logits, target)
        self.log('val_loss', val_loss, sync_dist=True, prog_bar=True, on_step=False, on_epoch=True)
        preds = logits.argmax(dim=-1)
        # print(preds)
        valid_targets = target != -100
        return {'loss': val_loss, 'preds': preds[valid_targets], 'target': target[valid_targets], 'pts': torch.cat(batch['pts'], axis=0)[valid_targets]}

    def convert_sync_batchnorm(self):
        self.model = ME.MinkowskiSyncBatchNorm.convert_sync_batchnorm(self.model)

    @staticmethod
    def add_argparse_args(parent_parser):
        parent_parser = BaseSegmentationModule.add_argparse_args(parent_parser)
        parser = parent_parser.add_argument_group("MinkSegModel")
        parser.add_argument("--use_seg_head", type=str2bool, nargs='?', const=True, default=False)
        parser.add_argument("--pretrained_minkunet_ckpt", type=str, default=None)
        parser.add_argument("--seg_head_in_bn", type=str2bool, nargs='?', const=True, default=False)
        parser.add_argument('--seg_head_dropout', type=float, default=0.3)
        parser.add_argument("--mlp_channels", type=str, default='1,4,8,4')
        parser.add_argument("--relative_mlp_channels", type=str2bool, nargs='?', const=True, default=True)
        parser.add_argument("--mlp_extra_in_channels", type=int, default=3)
        return parent_parser


