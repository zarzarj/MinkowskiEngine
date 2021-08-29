import torch
import torch.nn as nn
from torch.optim import SGD, Adam
from torch.optim.lr_scheduler import LambdaLR, StepLR

from pytorch_lightning.core import LightningModule
import MinkowskiEngine as ME
from examples.minkunet import MinkUNet34C, MinkUNet14A
from examples.minkunetodd import MinkUNet34C as MinkUNet34Codd
from examples.MeanAccuracy import MeanAccuracy
from examples.MeanIoU import MeanIoU
from pytorch_lightning.metrics import Accuracy, ConfusionMatrix, MetricCollection
from examples.MinkLightning import BaseSegmentationModule
from examples.str2bool import str2bool
from examples.basic_blocks import MLP
from examples.utils import interpolate_grid_feats

def to_precision(inputs, precision):
    # print(precision)
    if precision == 'mixed':
        dtype = torch.float16
    elif precision == 32:
        dtype = torch.float32
    elif precision == 64:
        dtype = torch.float64
    outputs = []
    for input in inputs:
        if isinstance(input, list):
            loutputs = []
            for loutput in input:
                if loutput is not None:
                    loutputs.append(loutput.to(dtype))
                else:
                    loutputs.append(loutput)
            outputs.append(loutputs)
            # print(loutputs)
        else:
            outputs.append(input.to(dtype))
    # print(outputs)
    return tuple(outputs)

class MinkowskiSegmentationModuleLIG(BaseSegmentationModule):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.mlp_channels = [int(i) for i in self.mlp_channels.split(',')]
        self.mlp_channels = (self.in_channels + self.mlp_extra_in_channels) * torch.tensor([1,4,8,4])
        if self.mink_sdf_to_seg:
            if self.odd_model:
                self.model = MinkUNet34Codd(self.in_channels, self.in_channels)
            else:
                self.model = MinkUNet34C(self.in_channels, self.in_channels)
        self.seg_head = nn.Sequential(MLP(self.mlp_channels, dropout=self.seg_head_dropout),
                                      nn.Conv1d(self.mlp_channels[-1], self.out_channels, kernel_size=1, bias=True)
                                      # nn.Linear(self.mlp_channels[-1], self.out_channels)
                                      )
        # print(self)

    def forward(self, x, pts, feats, rand_shift=None):
        bs = len(pts)
        # print(bs)
        if self.mink_sdf_to_seg:
            sparse_lats = self.model(x)
        else:
            sparse_lats = x
        if rand_shift is not None:
            list_of_coords, list_of_feats = sparse_lats.decomposed_coordinates_and_features
            for i in range(bs):
                list_of_coords[i] -= rand_shift[i]
            collated_coords, collated_feats = ME.utils.sparse_collate(list_of_coords,
                                                            list_of_feats,
                                                            dtype=x.dtype)
            new_sparse_lats = ME.SparseTensor(features=collated_feats.to(self.device), coordinates=collated_coords.int().to(self.device))
            seg_lats, min_coord, _ = new_sparse_lats.dense() # (b, *sizes, c)
            
        else:
            seg_lats, min_coord, _ = sparse_lats.dense() # (b, *sizes, c)
        
        seg_occ_in_list = []
        weights_list = []
        for i in range(bs):
            # print(pts[i].shape, feats[i].shape)
            lat, xloc = interpolate_grid_feats(pts[i], seg_lats[i].permute([1,2,3,0]), min_coord=min_coord) # (num_pts, 2**dim, c + 3), (num_pts, 2**dim)
            if feats[i] is not None:
                # print(feats[i].shape, lat.shape, xloc.shape)
                cur_seg_occ_in = torch.cat([lat, xloc, feats[i].unsqueeze(1).repeat(1,lat.shape[1],1)], dim=-1)
            else:
                cur_seg_occ_in = torch.cat([lat, xloc], dim=-1)
            # print(cur_seg_occ_in.shape, feats[i])
            cur_weights = 1 - torch.prod(torch.abs(xloc), axis=-1)
            seg_occ_in_list.append(cur_seg_occ_in)
            weights_list.append(cur_weights)
        seg_occ_in = torch.cat(seg_occ_in_list, dim=0).transpose(1,2) # (b x num_pts, c + 3, 2**dim)
        weights = torch.cat(weights_list, dim=0) # (b x num_pts, 2**dim)
        weights = weights.unsqueeze(dim=-1) # (b x num_pts, 2**dim, 1)
        if self.interpolate_grid_feats:
            weighted_feats = torch.bmm(seg_occ_in, weights) # (b x num_pts, c + 3, 1)
            logits = self.seg_head(weighted_feats).squeeze(dim=-1) # (b x num_pts, out_c, 1)
        else:
            seg_probs = self.seg_head(seg_occ_in) # (b x num_pts, out_c, 2**dim)
            logits = torch.bmm(seg_probs, weights).squeeze(dim=-1) # (b x num_pts, out_c)
        return logits

    # def on_after_backward(self):
    #     for k, v in self.named_parameters():
    #         print(k, v.grad)

    def training_step(self, batch, batch_idx):
        coords, lats, pts, feats, target = batch['coords'], batch['lats'], batch['pts'], batch['feats'], batch['labels']
        coords, lats, pts, feats = to_precision((coords, lats, pts, feats), self.trainer.precision)
        if self.trainer.datamodule.shift_coords:
            rand_shift = batch['rand_shift']
        else:
            rand_shift = None
        target = torch.cat(target, dim=0).long()
        # print('coords', coords.max(dim=0)[0])
        in_field = ME.TensorField(
            features=lats,
            coordinates=coords,
            quantization_mode=ME.SparseTensorQuantizationMode.UNWEIGHTED_AVERAGE,
            minkowski_algorithm=ME.MinkowskiAlgorithm.SPEED_OPTIMIZED,
            # minkowski_algorithm=ME.MinkowskiAlgorithm.MEMORY_EFFICIENT,
            device=self.device,
        )
        sinput = in_field.sparse()
        # print('coords2', sinput.coordinates.max(dim=0)[0])
        logits = self(sinput, pts, feats, rand_shift)
        train_loss = self.criterion(logits, target)

        self.log('train_loss', train_loss, sync_dist=True, prog_bar=True, on_step=False, on_epoch=True)
        preds = logits.argmax(dim=-1)
        valid_targets = target != -100

        if self.global_step % 10 == 0:
            torch.cuda.empty_cache()

        return {'loss': train_loss, 'preds': preds[valid_targets], 'target': target[valid_targets]}

    def validation_step(self, batch, batch_idx):
        coords, lats, pts, feats, target = batch['coords'], batch['lats'], batch['pts'], batch['feats'], batch['labels']
        coords, lats, pts, feats = to_precision((coords, lats, pts, feats), self.trainer.precision)
        target = torch.cat(target, dim=0).long()
        # print('coords', coords.max(dim=0)[0], [p.max(dim=0)[0] for p in pts], [p.min(dim=0)[0] for p in pts], batch['idxs'])
        in_field = ME.TensorField(
            features=lats,
            coordinates=coords,
            quantization_mode=ME.SparseTensorQuantizationMode.UNWEIGHTED_AVERAGE,
            minkowski_algorithm=ME.MinkowskiAlgorithm.SPEED_OPTIMIZED,
            device=self.device,
        )
        sinput = in_field.sparse()
        # print('coords2', sinput.coordinates.max(dim=0)[0])
        if self.global_step % 10 == 0:
            torch.cuda.empty_cache()
        logits = self(sinput, pts, feats)
        val_loss = self.criterion(logits, target)
        self.log('val_loss', val_loss, sync_dist=True, prog_bar=True, on_step=False, on_epoch=True)
        preds = logits.argmax(dim=-1)
        valid_targets = target != -100
        return {'loss': val_loss, 'preds': preds[valid_targets], 'target': target[valid_targets]}

    def convert_sync_batchnorm(self):
        if self.mink_sdf_to_seg:
            self.model = ME.MinkowskiSyncBatchNorm.convert_sync_batchnorm(self.model)

    @staticmethod
    def add_argparse_args(parent_parser):
        parent_parser = BaseSegmentationModule.add_argparse_args(parent_parser)
        parser = parent_parser.add_argument_group("MinkSegModelLIG")
        parser.add_argument("--interpolate_grid_feats", type=str2bool, nargs='?', const=True, default=False)
        parser.add_argument("--odd_model", type=str2bool, nargs='?', const=True, default=False)
        parser.add_argument("--mink_sdf_to_seg", type=str2bool, nargs='?', const=True, default=True)
        parser.add_argument('--seg_head_dropout', type=float, default=0.3)
        parser.add_argument("--mlp_channels", type=str, default='1,4,8,4')
        parser.add_argument("--mlp_extra_in_channels", type=int, default=3)
        return parent_parser

