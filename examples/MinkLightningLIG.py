import torch
import torch.nn as nn
from torch.optim import SGD, Adam
from torch.optim.lr_scheduler import LambdaLR, StepLR

from pytorch_lightning.core import LightningModule
import MinkowskiEngine as ME
from examples.minkunet import MinkUNet34C, MinkUNet14A
from examples.MeanAccuracy import MeanAccuracy
from examples.MeanIoU import MeanIoU
from pytorch_lightning.metrics import Accuracy, ConfusionMatrix, MetricCollection
from examples.MinkLightning import MinkowskiSegmentationModule

def act_layer(act, inplace=False, neg_slope=0.2, n_prelu=1):
    # activation layer
    act = act.lower()
    if act == 'relu':
        layer = nn.ReLU(inplace)
    elif act == 'leakyrelu':
        layer = nn.LeakyReLU(neg_slope, inplace)
    elif act == 'prelu':
        layer = nn.PReLU(num_parameters=n_prelu, init=neg_slope)
    else:
        raise NotImplementedError('activation layer [%s] is not found' % act)
    return layer

def norm_layer(norm_type, nc, track_running_stats=True):
    # normalization layer 1d
    norm = norm_type.lower()
    if norm == 'batch':
        layer = nn.BatchNorm1d(nc, affine=True, track_running_stats=track_running_stats)
    elif norm == 'layer':
        layer = nn.LayerNorm(nc, elementwise_affine=True)
    elif norm == 'instance':
        layer = nn.InstanceNorm1d(nc, affine=False)
    elif norm == 'none':
        layer = nn.Identity()
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm)
    return layer

class MLP(nn.Sequential):
    def __init__(self, mlp_channels, act='relu', norm='batch', bias=True, dropout=0.0):
        m = []
        in_channels = mlp_channels[0]
        for hidden_channels in mlp_channels[1:]:
            m.append(nn.Conv1d(in_channels, hidden_channels, kernel_size=1, bias=bias))
            if act is not None and act.lower() != 'none':
                m.append(act_layer(act))
            if norm is not None and norm.lower() != 'none':
                m.append(norm_layer(norm, hidden_channels))
            if dropout > 0:
                m.append(nn.Dropout2d(dropout, inplace=True))
            in_channels = hidden_channels
        super(MLP, self).__init__(*m)

        self.reset_parameters()

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d) or isinstance(m, nn.InstanceNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

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
            outputs.append([linput.to(dtype) for linput in input])
        else:
            outputs.append(input.to(dtype))
    return tuple(outputs)

class MinkowskiSegmentationModuleLIG(MinkowskiSegmentationModule):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.save_hyperparameters()
        self.mlp_channels = (self.in_channels + 3) * torch.tensor([1,2,4,2])
        self.model = MinkUNet34C(self.in_channels, self.mlp_channels[0] - 3)
        self.seg_head = nn.Sequential(MLP(self.mlp_channels),
                                      nn.Conv1d(self.mlp_channels[-1], self.out_channels, kernel_size=1, bias=True)
                                      # nn.Linear(self.mlp_channels[-1], self.out_channels)
                                      )
        # print(self)

    def forward(self, x, pts):
        # print(x)
        # print(x)
        seg_lats, _, _ = self.model(x).dense() # (b, *sizes, c)
        # print(seg_lats)
        bs = seg_lats.shape[0]
        seg_occ_in_list = []
        weights_list = []
        for i in range(bs):
            cur_seg_occ_in, cur_weights = get_implicit_feats(pts[i], seg_lats[i].permute([1,2,3,0])) # (num_pts, 2**dim, c + 3), (num_pts, 2**dim)
            seg_occ_in_list.append(cur_seg_occ_in)
            weights_list.append(cur_weights)
        seg_occ_in = torch.cat(seg_occ_in_list, dim=0).transpose(1,2) # (b x num_pts, c + 3, 2**dim)
        weights = torch.cat(weights_list, dim=0) # (b x num_pts, 2**dim)
        # print(weights.shape, seg_occ_in.shape, weights, seg_occ_in)
        seg_probs = self.seg_head(seg_occ_in) # (b x num_pts, out_c, 2**dim)
        weights = weights.unsqueeze(dim=-1) # (b x num_pts, 2**dim, 1)
        # print(weights.shape, seg_probs.shape)
        seg_weighted_probs = torch.bmm(seg_probs, weights).squeeze(dim=-1) # (b x num_pts, out_c)
        return seg_weighted_probs

    def training_step(self, batch, batch_idx):
        coords, feats, pts, target = batch['coords'], batch['feats'], batch['pts'], batch['labels']
        coords, feats, pts = to_precision((coords, feats, pts), self.trainer.precision)  
        target = torch.cat(target, dim=0)  
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
        logits = self(sinput, pts)
        # print(logits, target, logits.shape, target.shape)
        # print(logits, logits.argmax(-1))
        train_loss = self.criterion(logits, target)
        self.log('train_loss', train_loss, sync_dist=True, prog_bar=True, on_step=False, on_epoch=True)
        preds = logits.argmax(dim=-1)
        valid_targets = target != -100
        return {'loss': train_loss, 'preds': preds[valid_targets], 'target': target[valid_targets]}

    def validation_step(self, batch, batch_idx):
        coords, feats, pts, target = batch['coords'], batch['feats'], batch['pts'], batch['labels']
        coords, feats, pts = to_precision((coords, feats, pts), self.trainer.precision)
        target = torch.cat(target, dim=0) 
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
        logits = self(sinput, pts)
        val_loss = self.criterion(logits, target)
        self.log('val_loss', val_loss, sync_dist=True, prog_bar=True, on_step=False, on_epoch=True)
        preds = logits.argmax(dim=-1)
        valid_targets = target != -100
        return {'loss': val_loss, 'preds': preds[valid_targets], 'target': target[valid_targets]}

    @staticmethod
    def add_argparse_args(parent_parser):
        parent_parser = MinkowskiSegmentationModule.add_argparse_args(parent_parser)
        parser = parent_parser.add_argument_group("MinkSegModelLIG")
        return parent_parser

def get_implicit_feats(pts, grid, part_size=0.25):
    """Regular grid interpolator, returns inpterpolation coefficients.
    Args:
    pts: `[num_points, dim]` tensor, coordinates of points
    grid: `[b, *sizes, dim]` tensor of latents

    Returns:
    implicit feats: `[num_points, 2**dim * ( features + dim )]` tensor, neighbor
    latent codes and relative locations for each input point .
    """
    # get dimensions
    npts = pts.shape[0]
    dim = pts.shape[1]
    xmin = torch.min(pts, dim=0)[0] - part_size
    # normalize coords for interpolation
    pts -= xmin

    # find neighbor indices
    ind0 = torch.floor(2 * pts / part_size)-1  # `[num_points, dim]`
    # print(grid.shape, ind0.max(), ind0.min(), pts.max(dim=0), pts.min(dim=0))
    ind1 = torch.ceil(2 * pts / part_size)-1  # `[num_points, dim]`
    # print(ind0.max(dim=0), ind0.min(dim=0))
    # print(ind1.max(dim=0), ind1.min(dim=0))
    ind01 = torch.stack([ind0, ind1], dim=0)  # `[2, num_points, dim]`
    ind01 = torch.transpose(ind01, 1, 2)  # `[2, dim, num_points]`

    # generate combinations for enumerating neighbors
    com_ = torch.stack(torch.meshgrid(*tuple(torch.tensor([[0,1]] * dim))), dim=-1)
    com_ = torch.reshape(com_, [-1, dim])  # `[2**dim, dim]`
    # print(com_, com_.shape)
    dim_ = torch.reshape(torch.arange(0,dim), [1, -1])
    dim_ = torch.tile(dim_, [2**dim, 1])  # `[2**dim, dim]`
    # print(dim_, dim_.shape)
    gather_ind = torch.stack([com_, dim_], dim=-1)  # `[2**dim, dim, 2]`
    # print(gather_ind, gather_ind.shape)
    ind_ = gather_nd(ind01, gather_ind)  # [2**dim, dim, num_pts]
    # print(ind_, ind_.shape)
    ind_n = torch.transpose(ind_, 0,2).transpose(1,2)  # neighbor indices `[num_pts, 2**dim, dim]`
    # print(ind_n, ind_n.shape)
    # print(ind_n[...,0].max(), ind_n[...,1].max(), ind_n[...,2].max(), grid.shape)
    lat = gather_nd(grid, ind_n) # `[num_points, 2**dim, in_features]`
    # print(lat, lat.shape)

    # weights of neighboring nodes
    xyz0 = (ind0+1) * (part_size / 2)  # `[num_points, dim]`
    xyz1 = (ind0 + 2) * (part_size / 2)  # `[num_points, dim]`
    xyz01 = torch.stack([xyz0, xyz1], dim=-1)  # [num_points, dim, 2]`
    xyz01 = torch.transpose(xyz01, 0,2)  # [2, dim, num_points]
    # print(gather_ind.max(), gather_ind.min(), xyz01.shape)
    pos = gather_nd(xyz01, gather_ind)  # `[2**dim, dim, num_points]`
    pos = torch.transpose(pos, 0,2).transpose(1,2) # `[num_points, 2**dim, dim]`

    xloc = (torch.unsqueeze(pts, -2) - pos) # `[num_points, 2**dim, dim]`
    implicit_feats = torch.cat([lat, xloc], dim=-1) # `[num_points, 2**dim * (in_features + dim)]`

    dxyz_ = torch.abs(xloc) # `[num_points, 2**dim, dim]`
    weights = torch.prod(dxyz_, axis=-1)
    return implicit_feats, weights

def gather_nd(params, indices):
    orig_shape = list(indices.shape)
    num_samples = torch.prod(torch.tensor(orig_shape[:-1]))
    m = orig_shape[-1]
    n = len(params.shape)
    if m <= n:
        out_shape = orig_shape[:-1] + list(params.shape)[m:]
    else:
        raise ValueError(
            f'the last dimension of indices must less or equal to the rank of params. Got indices:{indices.shape}, params:{params.shape}. {m} > {n}'
        )

    # print(indices.shape, params.shape, indices.max(), indices.min())
    indices = indices.reshape((num_samples, m)).transpose(0, 1).tolist()
    # print(indices.shape, params.shape, indices.max(), indices.min())
    output = params[indices] 
    return output.reshape(out_shape).contiguous()