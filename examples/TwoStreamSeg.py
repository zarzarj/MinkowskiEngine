import torch
import torch.nn as nn

from examples.BaseSegLightning import BaseSegmentationModule
from examples.str2bool import str2bool

from examples.utils import interpolate_grid_feats, interpolate_sparsegrid_feats
import numpy as np
from examples.basic_blocks import MLP, norm_layer

import importlib

def get_obj_from_str(string):
    # From https://github.com/CompVis/taming-transformers
    module, cls = string.rsplit(".", 1)
    return getattr(importlib.import_module(module, package=None), cls)

class TwoStreamSegmentationModule(BaseSegmentationModule):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if self.color_backbone_class is not None:
            self.color_backbone = get_obj_from_str(self.color_backbone_class)(**kwargs,
                                            in_channels=self.feat_channels,
                                            out_channels=self.num_classes)
        if self.structure_backbone_class is not None:
            self.structure_backbone = get_obj_from_str(self.structure_backbone_class)(**kwargs,
                                            in_channels=self.feat_channels,
                                            out_channels=self.num_classes)

        # seg_feat_channels = self.color_backbone.feat_channels + self.structure_backbone.feat_channels
        seg_feat_channels = 96 * self.use_structure_feats + 128 * self.use_color_feats

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
        

    def forward(self, in_dict):
        if self.color_backbone_class is not None:
            color_logits, color_feats = self.color_backbone(in_dict, return_feats=True)
        else:
            color_feats = in_dict['color_feats']

        if self.structure_backbone_class is not None:
            structure_logits, structure_feats = self.structure_backbone(in_dict, return_feats=True)
        else:
            structure_feats = in_dict['structure_feats']
        
        fused_feats = []
        if self.use_color_feats:
            fused_feats.append(color_feats)
        if self.use_structure_feats:
            fused_feats.append(structure_feats)

        fused_feats = torch.cat(fused_feats, axis=1).unsqueeze(-1)
        # print(fused_feats.shape)

        logits = self.seg_head(fused_feats).squeeze(-1)

        return logits

    def convert_sync_batchnorm(self):
        pass

    @staticmethod
    def add_argparse_args(parent_parser):
        parent_parser = BaseSegmentationModule.add_argparse_args(parent_parser)
        parser = parent_parser.add_argument_group("ImplicitSegmentationModule")
        parser.add_argument("--color_backbone_class", type=str, default=None)
        parser.add_argument("--structure_backbone_class", type=str, default=None)

        parser.add_argument("--seg_head_in_bn", type=str2bool, nargs='?', const=True, default=False)
        parser.add_argument('--seg_head_dropout', type=float, default=0.3)
        parser.add_argument("--mlp_channels", type=str, default='1,4,8,4')
        parser.add_argument("--relative_mlp_channels", type=str2bool, nargs='?', const=True, default=True)
        parser.add_argument("--use_color_feats", type=str2bool, nargs='?', const=True, default=True)
        parser.add_argument("--use_structure_feats", type=str2bool, nargs='?', const=True, default=True)
        return parent_parser

