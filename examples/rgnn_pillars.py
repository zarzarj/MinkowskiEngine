from typing import Optional, List, NamedTuple
import copy

import numpy as np
import torch
from torch import Tensor
import torch.nn.functional as F
import torch.nn as nn
from torch.nn import ModuleList, Sequential, Linear, BatchNorm1d, ReLU, Dropout, Conv2d, Dropout2d

from torch_sparse import SparseTensor
from torch_geometric.nn import SAGEConv, GATConv, EdgeConv
import torch_geometric

from examples.EdgeConvs import EdgeConvZ, EdgeOnlyConv, DenseEdgeOnlyConv, EdgeOnlyConvPos, EffSparseEdgeConv, DenseEdgeConvTriplets, DenseEdgeConv, DenseMRConv
from examples.model_rev import RevGCN
from examples.MeanAccuracy import MeanAccuracy
from examples.MeanIoU import MeanIoU
from examples.str2bool import str2bool
from examples.BaseSegLightning import BaseSegmentationModule
from examples.basic_blocks import BasicGCNBlock, MLPLinear, MLP as basic_MLP
from pytorch_lightning.core import LightningModule


from torch_geometric.nn import PointConv, fps, radius, global_max_pool, knn_interpolate
from torch.nn import Sequential as Seq, Linear as Lin, ReLU, BatchNorm1d as BN
from torch_geometric.nn import DynamicEdgeConv

from examples.utils import save_pc

def to_precision(inputs, precision):
    if precision == 16:
        dtype = torch.float16
    elif precision == 32:
        dtype = torch.float32
    elif precision == 64:
        dtype = torch.float64
    return tuple([input.to(dtype) for input in inputs])

class LinearWrap(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(LinearWrap, self).__init__()
        self.in_channels = in_channels
        self.lin = Linear(in_channels, out_channels)

    def forward(self, x, edge_index):
        return self.lin(x)

class GCNWrap(nn.Module):
    def __init__(self, gcn, in_channels):
        super(GCNWrap, self).__init__()
        self.in_channels = in_channels
        self.gcn = gcn

    def forward(self, x, edge_index):
        # print(x.shape)
        return self.gcn(x, edge_index)

    def reset_parameters(self):
        self.gcn.reset_parameters()


class RevGNN_Rooms(LightningModule):
    def __init__(self, in_channels=3, out_channels=20, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        for name, value in kwargs.items():
            if name != "self":
                try:
                    setattr(self, name, value)
                except:
                    print(name, value)

        self.convs = ModuleList()
        if self.model == 'gat':
            in_conv = GATConv(in_channels, self.hidden_channels // self.heads, self.heads,
                              add_self_loops=False)
            back_conv = GATConv(self.hidden_channels // self.group, (self.hidden_channels // self.heads) // self.group,
                                self.heads, add_self_loops=False)
        elif self.model == 'edgeconv':
            self.edgeconv_channels = [int(i) for i in self.edgeconv_channels.split(',')]
            edgeconv_channels = [2 * in_channels] + self.edgeconv_channels
            conv_mlp = MLPLinear(mlp_channels=edgeconv_channels,
                                 norm=self.norm, dropout=self.dropout, 
                                 track_running_stats=self.track_running_stats,
                                 momentum=self.bn_momentum)
            in_conv = GCNWrap(EdgeConv(nn=conv_mlp, aggr=self.aggr), in_channels=in_channels)
            edgeconv_channels = np.array([2 * self.edgeconv_channels[-1]] + self.edgeconv_channels)
            conv_mlp = MLPLinear(mlp_channels=edgeconv_channels // self.group,
                                 norm=self.norm, dropout=self.dropout, 
                                 track_running_stats=self.track_running_stats,
                                 momentum=self.bn_momentum)
            back_conv = GCNWrap(EdgeConv(nn=conv_mlp, aggr=self.aggr), in_channels=(self.edgeconv_channels[-1]) // self.group)

        self.convs.append(BasicGCNBlock(in_conv, dropout=self.dropout, input_layer=True,
                                     track_running_stats=self.track_running_stats,
                                     momentum=self.bn_momentum))
        for i in range(self.num_layers - 1):
            if i == 0:
                cur_back_conv = back_conv
            else:
                cur_back_conv = copy.deepcopy(back_conv)
                cur_back_conv.reset_parameters()
            self.convs.append(BasicGCNBlock(cur_back_conv, dropout=self.dropout, norm=self.norm, 
                                         track_running_stats=self.track_running_stats,
                                         momentum=self.bn_momentum))

        if self.reversible:
            for i in range(1, self.num_layers):
                self.convs[i] = RevGCN(self.convs[i], group=self.group,
                                        preserve_rng_state=self.dropout>0)
        else:
            self.skips = ModuleList()
            self.skips.append(Linear(in_channels, self.hidden_channels))
            for _ in range(self.num_layers - 1):
                self.skips.append(Linear(self.hidden_channels, self.hidden_channels))

        # self.out = Linear(self.hidden_channels, out_channels, bias=True)
        self.mlp_channels = [self.hidden_channels] + [int(i) for i in self.mlp_channels.split(',')]
        self.mlp = Sequential(basic_MLP(mlp_channels=self.mlp_channels,
                                  norm=self.norm, dropout=self.dropout, 
                                         track_running_stats=self.track_running_stats,
                                         momentum=self.bn_momentum),
                              nn.Conv1d(self.mlp_channels[-1], out_channels, kernel_size=1, bias=True)
                              )
        # print(self)

    def forward(self, batch) -> Tensor:
        x = batch['feats']
        coords = batch['coords']
        adj = torch_geometric.nn.pool.knn_graph(x=coords[...,1:], k=16, batch=coords[...,0].long(),
                                                    loop=False, flow='source_to_target',
                                                    cosine=False)
        for i in range(len(self.convs)):
            x = self.forward_single(x, adj, i)
        return self.mlp(x.unsqueeze(-1)).squeeze(-1)

    def forward_single(self, x: Tensor, adj, cur_layer: int):
        out = self.convs[cur_layer](x, adj)
        if not self.reversible:
            out += self.skips[cur_layer](x)
        return out

    def convert_sync_batchnorm(self):
        return

    @staticmethod
    def add_argparse_args(parent_parser):
        parser = parent_parser.add_argument_group("RevGNNSegModel")
        # parser.add_argument("--in_channels", type=int, default=3)
        parser.add_argument("--hidden_channels", type=int, default=128)
        parser.add_argument("--heads", type=int, default=4)
        parser.add_argument("--num_layers", type=int, default=3)
        parser.add_argument("--group", type=int, default=2)
        parser.add_argument("--dropout", type=int, default=0.0)
        parser.add_argument("--bn_momentum", type=int, default=0.1)
        parser.add_argument("--norm", type=str, default='batch', choices=['batch', 'layer', 'instance', 'none'])
        parser.add_argument("--track_running_stats", type=str2bool, nargs='?', const=True, default=False)
        parser.add_argument("--reversible", type=str2bool, nargs='?', const=True, default=True)
        parser.add_argument("--model", type=str, default='gat', choices=['gat', 'edgeconv'])
        parser.add_argument("--aggr", type=str, default='max', choices=['max', 'mean', 'add'])
        parser.add_argument("--mlp_channels", type=str, default='512,256,128,64')
        parser.add_argument("--edgeconv_channels", type=str, default='128')
        return parent_parser


class PointwiseMLP(LightningModule):
    def __init__(self, in_channels=3, out_channels=20, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        for name, value in kwargs.items():
            if name != "self":
                try:
                    setattr(self, name, value)
                except:
                    print(name, value)

        self.mlp_channels = [in_channels] + [int(i) for i in self.mlp_channels.split(',')]
        self.mlp = Sequential(MLP(mlp_channels=self.mlp_channels,
                                  norm=self.norm, dropout=self.dropout,
                                  track_running_stats=self.track_running_stats,
                                  momentum=self.bn_momentum),
                              nn.Conv1d(self.mlp_channels[-1], out_channels, kernel_size=1, bias=True)
                              )
        # print(self)
        # print(self)

    def forward(self, batch) -> Tensor:
        x = batch['feats'].unsqueeze(-1)
        return self.mlp(x).squeeze(-1)

    def convert_sync_batchnorm(self):
        return

    @staticmethod
    def add_argparse_args(parent_parser):
        parser = parent_parser.add_argument_group("PointwiseMLPSegModel")
        # parser.add_argument("--in_channels", type=int, default=3)
        parser.add_argument("--dropout", type=int, default=0.0)
        parser.add_argument("--bn_momentum", type=int, default=0.1)
        parser.add_argument("--norm", type=str, default='batch', choices=['batch', 'layer', 'instance', 'none'])
        parser.add_argument("--track_running_stats", type=str2bool, nargs='?', const=True, default=False)
        parser.add_argument("--mlp_channels", type=str, default='256,128,64')
        return parent_parser


class STNkd(nn.Module):
    def __init__(self, k=64):
        super(STNkd, self).__init__()
        self.conv1 = torch.nn.Conv1d(k, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k*k)
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        # self.bn4 = nn.BatchNorm1d(512)
        # self.bn5 = nn.BatchNorm1d(256)
        self.bn4 = nn.Identity()
        self.bn5 = nn.Identity()
        self.k = k

    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        iden = torch.from_numpy(np.eye(self.k).flatten().astype(np.float32)).view(1,self.k*self.k).repeat(batchsize,1).type_as(x)
        x = x + iden
        x = x.view(-1, self.k, self.k)
        return x

class PointNetfeat(nn.Module):
    def __init__(self, in_channels=3, global_feat = True, feature_transform = False):
        super(PointNetfeat, self).__init__()
        self.stn = STNkd(in_channels)
        self.conv1 = torch.nn.Conv1d(in_channels, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.global_feat = global_feat
        self.feature_transform = feature_transform
        if self.feature_transform:
            self.fstn = STNkd(k=64)

    def forward(self, x):
        n_pts = x.size()[2]
        trans = self.stn(x)
        x = x.transpose(2, 1)
        x = torch.bmm(x, trans)
        x = x.transpose(2, 1)
        x = F.relu(self.bn1(self.conv1(x)))

        if self.feature_transform:
            trans_feat = self.fstn(x)
            x = x.transpose(2,1)
            x = torch.bmm(x, trans_feat)
            x = x.transpose(2,1)
        else:
            trans_feat = None

        pointfeat = x
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)
        if self.global_feat:
            return x, trans, trans_feat
        else:
            x = x.view(-1, 1024, 1).repeat(1, 1, n_pts)
            return torch.cat([x, pointfeat], 1), trans, trans_feat


class PointNetDenseCls(LightningModule):
    def __init__(self, in_channels=3, out_channels=20, feature_transform=False):
        super(PointNetDenseCls, self).__init__()
        self.feature_transform=feature_transform
        self.feat = PointNetfeat(in_channels=in_channels, global_feat=False, feature_transform=feature_transform)
        self.conv1 = torch.nn.Conv1d(1088, 512, 1)
        self.conv2 = torch.nn.Conv1d(512, 256, 1)
        self.conv3 = torch.nn.Conv1d(256, 128, 1)
        self.conv4 = torch.nn.Conv1d(128, out_channels, 1)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.bn3 = nn.BatchNorm1d(128)

    def forward(self, in_dict):
        x = in_dict['feats'].transpose(0,1).unsqueeze(0)

        # print(x.shape)
        batchsize = x.size()[0]
        n_pts = x.size()[2]
        x, trans, trans_feat = self.feat(x)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.conv4(x).squeeze(0).transpose(0,1)
        # print(x.shape)
        return x

    def convert_sync_batchnorm(self):
        return

    @staticmethod
    def add_argparse_args(parent_parser):
        parser = parent_parser.add_argument_group("PointNetDenseCls")
        return parent_parser



class SAModule(torch.nn.Module):
    def __init__(self, ratio, r, nn):
        super(SAModule, self).__init__()
        self.ratio = ratio
        self.r = r
        self.conv = PointConv(nn, add_self_loops=False)

    def forward(self, x, pos, batch):
        idx = fps(pos, batch, ratio=self.ratio)
        row, col = radius(pos, pos[idx], self.r, batch, batch[idx],
                          max_num_neighbors=64)
        edge_index = torch.stack([col, row], dim=0)
        x = self.conv(x, (pos, pos[idx]), edge_index)
        pos, batch = pos[idx], batch[idx]
        return x, pos, batch


class GlobalSAModule(torch.nn.Module):
    def __init__(self, nn):
        super(GlobalSAModule, self).__init__()
        self.nn = nn

    def forward(self, x, pos, batch):
        x = self.nn(torch.cat([x, pos], dim=1))
        x = global_max_pool(x, batch)
        pos = pos.new_zeros((x.size(0), 3))
        batch = torch.arange(x.size(0), device=batch.device)
        return x, pos, batch


def MLP(channels, batch_norm=True):
    return Seq(*[
        Seq(Lin(channels[i - 1], channels[i]), ReLU(), BN(channels[i]))
        for i in range(1, len(channels))
    ])

class FPModule(torch.nn.Module):
    def __init__(self, k, nn):
        super(FPModule, self).__init__()
        self.k = k
        self.nn = nn

    def forward(self, x, pos, batch, x_skip, pos_skip, batch_skip):
        x = knn_interpolate(x, pos, pos_skip, batch, batch_skip, k=self.k)
        if x_skip is not None:
            x = torch.cat([x, x_skip], dim=1)
        x = self.nn(x)
        return x, pos_skip, batch_skip

class PointNetv2(LightningModule):
    def __init__(self, in_channels=3, out_channels=20):
        super(PointNetv2, self).__init__()

        # Input channels account for both `pos` and node features.
        self.sa1_module = SAModule(0.2, 0.2, MLP([in_channels + 3, 64, 64, 128]))
        self.sa2_module = SAModule(0.25, 0.4, MLP([128 + 3, 128, 128, 256]))
        self.sa3_module = GlobalSAModule(MLP([256 + 3, 256, 512, 1024]))

        self.fp3_module = FPModule(1, MLP([1024 + 256, 256, 256]))
        self.fp2_module = FPModule(3, MLP([256 + 128, 256, 128]))
        self.fp1_module = FPModule(3, MLP([128 + in_channels, 128, 128, 128]))

        self.lin1 = torch.nn.Linear(128, 128)
        self.lin2 = torch.nn.Linear(128, 128)
        self.lin3 = torch.nn.Linear(128, out_channels)

    def forward(self, batch):
        # batch_0_idx = batch['coords'][:,0] == 0
        # save_pc(batch['coords'][batch_0_idx,1:].cpu().numpy(), batch['feats'][batch_0_idx,:3].cpu().numpy(), 'test_s3dis_fwd.ply')
        # assert(True == False)
        # print(batch['feats'], batch['feats'].shape)
        sa0_out = (batch['feats'], batch['coords'][:,1:], batch['coords'][:,0].long())
        sa1_out = self.sa1_module(*sa0_out)
        sa2_out = self.sa2_module(*sa1_out)
        sa3_out = self.sa3_module(*sa2_out)

        fp3_out = self.fp3_module(*sa3_out, *sa2_out)
        fp2_out = self.fp2_module(*fp3_out, *sa1_out)
        x, _, _ = self.fp1_module(*fp2_out, *sa0_out)

        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin3(x)
        return x
        
    def convert_sync_batchnorm(self):
        return

    @staticmethod
    def add_argparse_args(parent_parser):
        parser = parent_parser.add_argument_group("PointNetv2")
        return parent_parser


class DGCNN(LightningModule):
    def __init__(self, in_channels=3, out_channels=20, k=30, aggr='max'):
        super(DGCNN, self).__init__()

        self.conv1 = DynamicEdgeConv(MLP([2 * (in_channels+3), 64, 64]), k, aggr)
        self.conv2 = DynamicEdgeConv(MLP([2 * 64, 64, 64]), k, aggr)
        self.conv3 = DynamicEdgeConv(MLP([2 * 64, 64, 64]), k, aggr)
        self.lin1 = MLP([3 * 64, 1024])

        self.mlp = Seq(MLP([1024, 256]), Dropout(0.5), MLP([256, 128]),
                       Dropout(0.5), Lin(128, out_channels))

    def forward(self, data):
        x, pos, batch = (data['feats'], data['coords'][:,1:], data['coords'][:,0].long())
        x0 = torch.cat([x, pos], dim=-1)
        x1 = self.conv1(x0, batch)
        x2 = self.conv2(x1, batch)
        x3 = self.conv3(x2, batch)
        out = self.lin1(torch.cat([x1, x2, x3], dim=1))
        out = self.mlp(out)
        return F.log_softmax(out, dim=1)

    def convert_sync_batchnorm(self):
        return
        
    @staticmethod
    def add_argparse_args(parent_parser):
        parser = parent_parser.add_argument_group("DGCNN")
        return parent_parser

def knn(x, k):
    inner = -2*torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x**2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)
 
    idx = pairwise_distance.topk(k=k, dim=-1)[1]   # (batch_size, num_points, k)
    return idx

def get_graph_feature(x, k=20, idx=None, use_coords=False):
    batch_size = x.size(0)
    num_points = x.size(2)
    # x = x.view(batch_size, -1, num_points)
    if idx is None:
        if use_coords:
            idx = knn(x[:, :3], k=k)
        else:
            idx = knn(x, k=k)   # (batch_size, num_points, k)
            
    device = torch.device('cuda')

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1)*num_points

    idx = idx + idx_base

    idx = idx.view(-1)
 
    _, num_dims, _ = x.size()

    x = x.transpose(2, 1).contiguous()   # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims) #   batch_size * num_points * k + range(0, batch_size*num_points)
    feature = x.view(batch_size*num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims) 
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)
    
    feature = torch.cat((feature-x, x), dim=3).permute(0, 3, 1, 2).contiguous()
  
    return feature      # (batch_size, 2*num_dims, num_points, k)

class DGCNN_semseg(nn.Module):
    def __init__(self, in_channels=3, out_channels=20, **kwargs):
        super(DGCNN_semseg, self).__init__()
        for name, value in kwargs.items():
            if name != "self":
                try:
                    setattr(self, name, value)
                except:
                    print(name, value)
        self.out_channels = out_channels
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(64)
        self.bn4 = nn.BatchNorm2d(64)
        self.bn5 = nn.BatchNorm2d(64)
        self.bn6 = nn.BatchNorm1d(self.emb_dims)
        self.bn7 = nn.BatchNorm1d(512)
        self.bn8 = nn.BatchNorm1d(256)

        self.conv1 = nn.Sequential(nn.Conv2d((3 + in_channels) * 2, 64, kernel_size=1, bias=False),
                                   self.bn1,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1, bias=False),
                                   self.bn2,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv3 = nn.Sequential(nn.Conv2d(64*2, 64, kernel_size=1, bias=False),
                                   self.bn3,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv4 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1, bias=False),
                                   self.bn4,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv5 = nn.Sequential(nn.Conv2d(64*2, 64, kernel_size=1, bias=False),
                                   self.bn5,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv6 = nn.Sequential(nn.Conv1d(192, self.emb_dims, kernel_size=1, bias=False),
                                   self.bn6,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv7 = nn.Sequential(nn.Conv1d(1216, 512, kernel_size=1, bias=False),
                                   self.bn7,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv8 = nn.Sequential(nn.Conv1d(512, 256, kernel_size=1, bias=False),
                                   self.bn8,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.dp1 = nn.Dropout(p=self.dropout)
        self.conv9 = nn.Conv1d(256, out_channels, kernel_size=1, bias=False)
        

    def forward(self, batch):
        x = torch.cat([batch['coords'].transpose(1,2), batch['feats']], axis=1)
        batch_size = x.size(0)
        num_points = x.size(2)
        x = get_graph_feature(x, k=self.k, use_coords=True)   # (batch_size, 9, num_points) -> (batch_size, 9*2, num_points, k)
        x = self.conv1(x)                       # (batch_size, 9*2, num_points, k) -> (batch_size, 64, num_points, k)
        x = self.conv2(x)                       # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points, k)
        x1 = x.max(dim=-1, keepdim=False)[0]    # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)

        x = get_graph_feature(x1, k=self.k)     # (batch_size, 64, num_points) -> (batch_size, 64*2, num_points, k)
        x = self.conv3(x)                       # (batch_size, 64*2, num_points, k) -> (batch_size, 64, num_points, k)
        x = self.conv4(x)                       # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points, k)
        x2 = x.max(dim=-1, keepdim=False)[0]    # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)

        x = get_graph_feature(x2, k=self.k)     # (batch_size, 64, num_points) -> (batch_size, 64*2, num_points, k)
        x = self.conv5(x)                       # (batch_size, 64*2, num_points, k) -> (batch_size, 64, num_points, k)
        x3 = x.max(dim=-1, keepdim=False)[0]    # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)

        x = torch.cat((x1, x2, x3), dim=1)      # (batch_size, 64*3, num_points)

        x = self.conv6(x)                       # (batch_size, 64*3, num_points) -> (batch_size, emb_dims, num_points)
        x = x.max(dim=-1, keepdim=True)[0]      # (batch_size, emb_dims, num_points) -> (batch_size, emb_dims, 1)

        x = x.repeat(1, 1, num_points)          # (batch_size, 1024, num_points)
        x = torch.cat((x, x1, x2, x3), dim=1)   # (batch_size, 1024+64*3, num_points)

        x = self.conv7(x)                       # (batch_size, 1024+64*3, num_points) -> (batch_size, 512, num_points)
        x = self.conv8(x)                       # (batch_size, 512, num_points) -> (batch_size, 256, num_points)
        x = self.dp1(x)
        x = self.conv9(x)                       # (batch_size, 256, num_points) -> (batch_size, 13, num_points)
        # print(x.shape)
        return x.transpose(1,2).contiguous().reshape(-1, self.out_channels)

    def convert_sync_batchnorm(self):
        return
        
    @staticmethod
    def add_argparse_args(parent_parser):
        parser = parent_parser.add_argument_group("DGCNN_semseg")
        parser = parent_parser.add_argument('--emb_dims', default=1024, type=int)
        parser = parent_parser.add_argument('--k', default=20, type=int)
        parser = parent_parser.add_argument('--dropout', default=0.5, type=float, help='ratio of dropout')
        return parent_parser