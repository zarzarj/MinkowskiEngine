import torch
import torch.nn as nn
from torch.optim import SGD, Adam
from torch.optim.lr_scheduler import LambdaLR, StepLR

from pytorch_lightning.core import LightningModule
import MinkowskiEngine as ME
from examples.minkunet_sparse import MinkUNet34C, MinkUNet14A, MinkUNet34CShallow
# from examples.minkunetodd import MinkUNet34C as MinkUNet34Codd
from examples.BaseSegLightning import BaseSegmentationModule
from examples.str2bool import str2bool
from examples.basic_blocks import MLP, norm_layer
from examples.utils import interpolate_grid_feats, interpolate_sparsegrid_feats
import numpy as np

class MinkowskiSegmentationModuleLIG(BaseSegmentationModule):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if self.mink_sdf_to_seg:
            self.model = MinkUNet34C(self.feat_channels, self.feat_channels)
                
        self.mlp_channels = [int(i) for i in self.mlp_channels.split(',')]
        if self.relative_mlp_channels:
            self.mlp_channels = (self.seg_feat_channels) * np.array(self.mlp_channels)
            # print(self.mlp_channels)
        else:
            self.mlp_channels = [self.seg_feat_channels] + self.mlp_channels

        seg_head_list = []
        if self.seg_head_in_bn:
            seg_head_list.append(norm_layer(norm_type='batch', nc=self.mlp_channels[0]))
        seg_head_list += [MLP(self.mlp_channels, dropout=self.seg_head_dropout),
                         nn.Conv1d(self.mlp_channels[-1], self.num_classes, kernel_size=1, bias=True)]
        self.seg_head = nn.Sequential(*seg_head_list)

        if self.pretrained_minkunet_ckpt is not None:
            # print(self.model)
            pretrained_ckpt = torch.load(self.pretrained_minkunet_ckpt)
            if 'state_dict' in pretrained_ckpt:
                pretrained_ckpt = pretrained_ckpt['state_dict']
            # print(pretrained_ckpt['state_dict'].keys())
            del pretrained_ckpt['conv0p1s1.kernel']
            del pretrained_ckpt['final.kernel']
            del pretrained_ckpt['final.bias']
            # print(pretrained_ckpt)
            self.model.load_state_dict(pretrained_ckpt, strict=False)
        # print(self)

    def forward(self, batch):
        pts = batch['pts']
        lats = batch['feats']
        coords = batch['coords']
        feats = batch['seg_feats']
        in_field = ME.TensorField(
            features=lats,
            coordinates=coords,
            quantization_mode=ME.SparseTensorQuantizationMode.UNWEIGHTED_AVERAGE,
            minkowski_algorithm=ME.MinkowskiAlgorithm.SPEED_OPTIMIZED,
            # minkowski_algorithm=ME.MinkowskiAlgorithm.MEMORY_EFFICIENT,
            device=self.device,
        )
        x = in_field.sparse()

        bs = len(pts)
        if self.mink_sdf_to_seg:
            sparse_lats = self.model(x)
        else:
            sparse_lats = x

        # seg_occ_in_list = []
        # weights_list = []
        logits_list = []
        for i in range(bs):
            lat, xloc, weights = interpolate_sparsegrid_feats(pts[i], sparse_lats.coordinates_at(batch_index=i),
                                                                   sparse_lats.features_at(batch_index=i),
                                                                   overlap_factor=self.overlap_factor) # (num_pts, 2**dim, c), (num_pts, 2**dim, 3)
            if self.interpolate_grid_feats and self.average_xlocs:
                xloc = xloc.mean(axis=1, keepdim=True).repeat(1, lat.shape[1], 1)
            if feats[i] is not None:
                seg_occ_in = torch.cat([lat, xloc, feats[i].unsqueeze(1).repeat(1,lat.shape[1],1)], dim=-1)
            else:
                seg_occ_in = torch.cat([lat, xloc], dim=-1)

            weights = weights.unsqueeze(dim=-1)
            seg_occ_in = seg_occ_in.transpose(1,2)
            # print(weights.shape, seg_occ_in.shape)
            if self.interpolate_grid_feats:
                weighted_feats = torch.bmm(seg_occ_in, weights) # (num_pts, c + 3, 1)
                logits = self.seg_head(weighted_feats).squeeze(dim=-1) # (num_pts, out_c, 1)
            else:
                seg_probs = self.seg_head(seg_occ_in) # (num_pts, out_c, 2**dim)
                logits = torch.bmm(seg_probs, weights).squeeze(dim=-1) # (num_pts, out_c)
            logits_list.append(logits)
        #     seg_occ_in_list.append(cur_seg_occ_in)
        #     weights_list.append(weights)
        # seg_occ_in = torch.cat(seg_occ_in_list, dim=0).transpose(1,2) # (b x num_pts, c + 3, 2**dim)
        # weights = torch.cat(weights_list, dim=0) # (b x num_pts, 2**dim)
        # weights = weights.unsqueeze(dim=-1) # (b x num_pts, 2**dim, 1)
        logits = torch.cat(logits_list, dim=0) # (b x num_pts, out_c)
        return logits

    def convert_sync_batchnorm(self):
        if self.mink_sdf_to_seg:
            self.model = ME.MinkowskiSyncBatchNorm.convert_sync_batchnorm(self.model)

    @staticmethod
    def add_argparse_args(parent_parser):
        parent_parser = BaseSegmentationModule.add_argparse_args(parent_parser)
        parser = parent_parser.add_argument_group("MinkSegModelLIG")
        parser.add_argument("--interpolate_grid_feats", type=str2bool, nargs='?', const=True, default=False)
        parser.add_argument("--average_xlocs", type=str2bool, nargs='?', const=True, default=False)
        parser.add_argument("--pretrained_minkunet_ckpt", type=str, default=None)
        parser.add_argument("--shallow_model", type=str2bool, nargs='?', const=True, default=False)
        parser.add_argument("--mink_sdf_to_seg", type=str2bool, nargs='?', const=True, default=True)
        
        parser.add_argument("--seg_head_in_bn", type=str2bool, nargs='?', const=True, default=False)
        parser.add_argument('--seg_head_dropout', type=float, default=0.3)
        parser.add_argument("--mlp_channels", type=str, default='1,4,8,4')
        parser.add_argument("--relative_mlp_channels", type=str2bool, nargs='?', const=True, default=True)
        parser.add_argument("--mlp_extra_in_channels", type=int, default=3)
        parser.add_argument("--overlap_factor", type=int, default=2)
        return parent_parser

