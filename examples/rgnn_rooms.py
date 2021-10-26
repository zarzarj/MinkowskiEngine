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
from examples.basic_blocks import MLP, BasicGCNBlock
from pytorch_lightning.core import LightningModule

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
        if self.model == 'rgat':
            in_conv = GATConv(in_channels, self.hidden_channels // self.heads, self.heads,
                              add_self_loops=False)
            back_conv = GATConv(self.hidden_channels // self.group, (self.hidden_channels // self.heads) // self.group,
                                self.heads, add_self_loops=False)

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
        self.mlp = Sequential(MLP(mlp_channels=self.mlp_channels,
                                  norm=self.norm, dropout=self.dropout, 
                                         track_running_stats=self.track_running_stats,
                                         momentum=self.bn_momentum),
                              nn.Conv1d(self.mlp_channels[-1], out_channels, kernel_size=1, bias=True)
                              )
        # print(self)

    def forward(self, batch, return_feats=False) -> Tensor:
        # print(coords, coords.max(), coords.min())
        # coords = batch['coords']
        x = batch['feats']
        adj = batch['adj']
        # print(x)
        # print(adj)
        # print(x.shape)
        # adj = torch_geometric.nn.pool.knn_graph(x=coords[...,1:], k=16, batch=coords[...,0].long(),
        #                                             loop=False, flow='source_to_target',
        #                                             cosine=False)

        for i in range(len(self.convs)):

            # print(x.shape)
            x = self.forward_single(x, adj, i)
        # x = x.unsqueeze(-1)
        # print(x.shape, coords.shape)
        if return_feats:
            return self.mlp(x.unsqueeze(-1)).squeeze(-1), x
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
        parser.add_argument("--model", type=str, default='rgat', choices=['rgat'])
        parser.add_argument("--mlp_channels", type=str, default='256,128,64')
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