import torch
import torch.nn as nn

from examples.BaseSegLightning import BaseSegmentationModule
from examples.str2bool import str2bool

from examples.utils import interpolate_grid_feats, interpolate_sparsegrid_feats
import numpy as np
from examples.basic_blocks import MLP, norm_layer

import importlib
import argparse

import MinkowskiEngine as ME

def get_obj_from_str(string):
    # From https://github.com/CompVis/taming-transformers
    module, cls = string.rsplit(".", 1)
    return getattr(importlib.import_module(module, package=None), cls)

class TwoStreamSegmentationModule(BaseSegmentationModule):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.loss_weights = []
        if self.use_fused_feats:
            self.loss_weights.append(1)
        if self.color_backbone_class is not None:
            self.color_backbone = get_obj_from_str(self.color_backbone_class)
            self.color_backbone = self.color_backbone(**self.color_backbone_args,
                                            in_channels=self.feat_channels,
                                            out_channels=self.num_classes)
            color_feats = self.color_backbone.hidden_channels
            self.loss_weights.append(1)
        else:
            color_feats = 128

        if self.structure_backbone_class is not None:
            # print(self.feat_channels)
            self.structure_backbone = get_obj_from_str(self.structure_backbone_class)
            self.structure_backbone = self.structure_backbone(**self.structure_backbone_args,
                                            in_channels=self.feat_channels,
                                            out_channels=self.num_classes)
            self.loss_weights.append(1)


        if self.use_fused_feats:
            # seg_feat_channels = self.color_backbone.feat_channels + self.structure_backbone.feat_channels
            seg_feat_channels = 96 * self.use_structure_feats + color_feats * self.use_color_feats
            self.mlp_channels = [int(i) for i in self.mlp_channels.split(',')]
            if self.relative_mlp_channels:
                self.mlp_channels = (seg_feat_channels) * np.array(self.mlp_channels)
            else:
                self.mlp_channels = [seg_feat_channels] + self.mlp_channels
            seg_head_list = []
            if self.seg_head_in_bn:
                seg_head_list.append(norm_layer(norm_type='batch', nc=self.mlp_channels[0]))
            seg_head_list += [MLP(self.mlp_channels, dropout=self.seg_head_dropout),
                             nn.Conv1d(self.mlp_channels[-1], self.num_classes, kernel_size=1, bias=True)]
            self.seg_head = nn.Sequential(*seg_head_list)

        self.loss_weights = torch.tensor(self.loss_weights, requires_grad=False)
        self.base_criterion = self.criterion
        self.criterion = MultiStreamLoss(self.base_criterion, self.loss_weights)
        if self.gradient_blend_frequency != -1:
            self.last_val_loss = torch.zeros_like(self.loss_weights)
            self.current_val_loss = torch.zeros_like(self.loss_weights)
            self.last_train_loss = torch.zeros_like(self.loss_weights)
        

    def forward(self, in_dict):
        if self.color_backbone_class is not None:
            color_logits, color_feats = self.color_backbone(in_dict, return_feats=True)
        else:
            color_feats = in_dict['color_feats']

        if self.structure_backbone_class is not None:
            structure_logits, structure_feats = self.structure_backbone(in_dict, return_feats=True)
        else:
            structure_feats = in_dict['structure_feats']

        logits = []
        if self.use_fused_feats:
            fused_feats = []
            if self.use_color_feats:
                fused_feats.append(color_feats)
            if self.use_structure_feats:
                fused_feats.append(structure_feats)
            fused_feats = torch.cat(fused_feats, axis=1).unsqueeze(-1)
            logits.append(self.seg_head(fused_feats).squeeze(-1))
        if self.color_backbone_class is not None:
            logits.append(color_logits)
        if self.structure_backbone_class is not None:
            logits.append(structure_logits)
        # print(logits)
        return torch.stack(logits)

    def convert_sync_batchnorm(self):
        if self.structure_backbone is not None:
            self.structure_backbone = ME.MinkowskiSyncBatchNorm.convert_sync_batchnorm(self.structure_backbone)

    def training_epoch_end(self, training_step_outputs):
        if self.gradient_blend_frequency != -1 and self.trainer.current_epoch % self.gradient_blend_frequency == 0:
            epoch_losses = torch.tensor([[loss.detach().cpu() for loss in pred['losses']] for pred in training_step_outputs]).mean(axis=0, keepdim=False)
            delta_G_Nn = self.last_val_loss - self.current_val_loss
            O_Nn = self.current_val_loss - epoch_losses
            O_N = self.last_val_loss - self.last_train_loss
            delta_O_Nn = O_Nn - O_N
            if self.trainer.current_epoch != 0:
                self.loss_weights = (delta_G_Nn/(delta_O_Nn**2)).abs()
                self.loss_weights /= self.loss_weights.sum()
                # print(self.loss_weights)
            self.last_train_loss = epoch_losses


    def validation_epoch_end(self, validation_step_outputs):
        if self.gradient_blend_frequency != -1 and self.trainer.current_epoch % self.gradient_blend_frequency == 0:
            epoch_losses = torch.tensor([[loss.detach().cpu() for loss in pred['losses']] for pred in validation_step_outputs]).mean(axis=0, keepdim=False)
            self.last_val_loss = self.current_val_loss
            self.current_val_loss = epoch_losses
    

    @staticmethod
    def add_argparse_args(parent_parser):
        parent_parser = BaseSegmentationModule.add_argparse_args(parent_parser)
        parser = parent_parser.add_argument_group("ImplicitSegmentationModule")
        parser.add_argument("--seg_head_in_bn", type=str2bool, nargs='?', const=True, default=False)
        parser.add_argument('--seg_head_dropout', type=float, default=0.3)
        parser.add_argument("--mlp_channels", type=str, default='512,256,128,64')
        parser.add_argument("--relative_mlp_channels", type=str2bool, nargs='?', const=True, default=False)
        parser.add_argument("--use_color_feats", type=str2bool, nargs='?', const=True, default=True)
        parser.add_argument("--use_structure_feats", type=str2bool, nargs='?', const=True, default=True)
        parser.add_argument("--use_fused_feats", type=str2bool, nargs='?', const=True, default=True)
        parser.add_argument("--gradient_blend_frequency", type=int, default=-1)
        return parent_parser


class MultiStreamLoss(torch.nn.Module):
    def __init__(self, loss_fn, loss_weights):
        super().__init__()
        self.loss_fn = loss_fn
        self.loss_weights = loss_weights

    def forward(self, multi_logits, targets):
        loss = 0
        losses = []
        # print(multi_logits.shape, targets.shape)
        for i, logits in enumerate(multi_logits):
            # print(logits.shape, targets.shape)
            losses.append(self.loss_fn(logits, targets) * self.loss_weights[i])
            loss += losses[-1]
            # print(loss)
        return {"loss":loss, "losses": [l.detach() for l in losses]}