import torch
import torch.nn as nn

from examples.BaseSegLightning import BaseSegmentationModule
from examples.str2bool import str2bool

from examples.utils import interpolate_grid_feats, interpolate_sparsegrid_feats
import numpy as np
from examples.basic_blocks import MLP, norm_layer

class ImplicitSegmentationModule(BaseSegmentationModule):
    def __init__(self, backbone, **kwargs):
        super().__init__(**kwargs)
        # print(self.backbone)
        self.backbone=backbone
        # print(self.backbone)
        if self.sdf_to_seg:
            # self.backbone = get_obj_from_str(self.backbone)(self.in_channels, self.in_channels)
            if self.pretrained_backbone_ckpt is not None:
                pretrained_ckpt = torch.load(self.pretrained_backbone_ckpt)
                if 'state_dict' in pretrained_ckpt:
                    pretrained_ckpt = pretrained_ckpt['state_dict']
                del pretrained_ckpt['conv0p1s1.kernel']
                del pretrained_ckpt['final.kernel']
                del pretrained_ckpt['final.bias']
                self.model.load_state_dict(pretrained_ckpt, strict=False)

        self.mlp_channels = [int(i) for i in self.mlp_channels.split(',')]
        if self.relative_mlp_channels:
            self.mlp_channels = (self.seg_in_channels) * np.array(self.mlp_channels)
            # print(self.mlp_channels)
        else:
            self.mlp_channels = [self.seg_in_channels] + self.mlp_channels
        seg_head_list = []
        if self.seg_head_in_bn:
            seg_head_list.append(norm_layer(norm_type='batch', nc=self.mlp_channels[0]))
        seg_head_list += [MLP(self.mlp_channels, dropout=self.seg_head_dropout),
                         nn.Conv1d(self.mlp_channels[-1], self.num_classes, kernel_size=1, bias=True)]
        self.seg_head = nn.Sequential(*seg_head_list)
        

    def forward(self, in_dict):
        if self.sdf_to_seg:
            seg_lats = self.backbone(in_dict)
        else:
            seg_lats = in_dict['lats']

        # print(seg_lats.shape)

        bs = len(in_dict['pts'])
        logits_list = []
        for i in range(bs):
            if in_dict['rand_shift'][i] is not None:
                cur_idx = in_dict['coords'][...,0] == i
                cur_coords = in_dict['coords'][cur_idx][...,1:] -  in_dict['rand_shift'][i]
                cur_lats = seg_lats[cur_idx]
            # print(seg_lats.shape, in_dict['pts'][i].shape)
            lat, xloc, weights = interpolate_sparsegrid_feats(in_dict['pts'][i], cur_coords.long(), cur_lats,
                                                                   overlap_factor=self.overlap_factor) # (num_pts, 2**dim, c), (num_pts, 2**dim, 3)
            if self.interpolate_grid_feats and self.average_xlocs:
                xloc = xloc.mean(axis=1, keepdim=True).repeat(1, lat.shape[1], 1)
            if in_dict['feats'][i] is not None:
                seg_occ_in = torch.cat([lat, xloc, in_dict['feats'][i].unsqueeze(1).repeat(1,lat.shape[1],1)], dim=-1)
            else:
                seg_occ_in = torch.cat([lat, xloc], dim=-1)
            weights = weights.unsqueeze(dim=-1)
            seg_occ_in = seg_occ_in.transpose(1,2)
            if self.interpolate_grid_feats:
                weighted_feats = torch.bmm(seg_occ_in, weights) # (num_pts, c + 3, 1)
                logits = self.seg_head(weighted_feats).squeeze(dim=-1) # (num_pts, out_c, 1)
            else:
                seg_probs = self.seg_head(seg_occ_in) # (num_pts, out_c, 2**dim)
                logits = torch.bmm(seg_probs, weights).squeeze(dim=-1) # (num_pts, out_c)
            logits_list.append(logits)
        logits = torch.cat(logits_list, dim=0) # (b x num_pts, out_c)
        return logits

    def convert_sync_batchnorm(self):
        self.backbone.convert_sync_batchnorm()

    @staticmethod
    def add_argparse_args(parent_parser):
        parent_parser = BaseSegmentationModule.add_argparse_args(parent_parser)
        parser = parent_parser.add_argument_group("ImplicitSegmentationModule")
        parser.add_argument("--interpolate_grid_feats", type=str2bool, nargs='?', const=True, default=False)
        parser.add_argument("--average_xlocs", type=str2bool, nargs='?', const=True, default=False)
        parser.add_argument("--pretrained_backbone_ckpt", type=str, default=None)
        parser.add_argument("--sdf_to_seg", type=str2bool, nargs='?', const=True, default=True)
        
        parser.add_argument("--seg_head_in_bn", type=str2bool, nargs='?', const=True, default=False)
        parser.add_argument('--seg_head_dropout', type=float, default=0.3)
        parser.add_argument("--mlp_channels", type=str, default='1,4,8,4')
        parser.add_argument("--relative_mlp_channels", type=str2bool, nargs='?', const=True, default=True)
        parser.add_argument("--mlp_extra_in_channels", type=int, default=3)
        parser.add_argument("--overlap_factor", type=int, default=2)
        # parser.add_argument("--latent_model", type=str, default='examples.minkunet.MinkUNet34C')
        return parent_parser

