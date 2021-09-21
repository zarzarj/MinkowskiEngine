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

from examples.EdgeConvs import EdgeConvZ, EdgeOnlyConv, DenseEdgeOnlyConv, EdgeOnlyConvPos, EffSparseEdgeConv, DenseEdgeConvTriplets, DenseEdgeConv, DenseMRConv
from examples.model_rev import RevGCN
from examples.MeanAccuracy import MeanAccuracy
from examples.MeanIoU import MeanIoU
from examples.str2bool import str2bool
from examples.BaseSegLightning import BaseSegmentationModule
from examples.basic_blocks import MLP, BasicGCNBlock

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


class RevGNN_Rooms(BaseSegmentationModule):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.convs = ModuleList()
        if self.model == 'rgat':
            in_conv = GATConv(self.in_channels, self.hidden_channels // self.heads, self.heads,
                              add_self_loops=False)
            back_conv = GATConv(self.hidden_channels // self.group, (self.hidden_channels // self.heads) // self.group,
                                self.heads, add_self_loops=False)

        self.convs.append(BasicGCNBlock(in_conv, dropout=self.dropout, input_layer=True,
                                     track_running_stats=self.track_running_stats))
        for i in range(self.num_layers - 1):
            if i == 0:
                cur_back_conv = back_conv
            else:
                cur_back_conv = copy.deepcopy(back_conv)
                cur_back_conv.reset_parameters()
            self.convs.append(BasicGCNBlock(cur_back_conv, dropout=self.dropout, norm=self.norm, 
                                         track_running_stats=self.track_running_stats))

        if self.reversible:
            for i in range(1, self.num_layers):
                self.convs[i] = RevGCN(self.convs[i], group=self.group,
                                        preserve_rng_state=self.dropout>0)
        else:
            self.skips = ModuleList()
            self.skips.append(Linear(self.in_channels, self.hidden_channels))
            for _ in range(self.num_layers - 1):
                self.skips.append(Linear(self.hidden_channels, self.hidden_channels))

        self.mlp_channels = [self.in_channels] + [int(i) for i in self.mlp_channels.split(',')]
        self.mlp = Sequential(MLP(mlp_channels=self.mlp_channels,
                                  norm=self.norm, dropout=self.dropout),
                              Linear(self.mlp_channels[-1], self.out_channels, bias=True)
                              )

    def forward(self, x: Tensor, adj) -> Tensor:
        for i in range(len(self.convs)):
            x = self.forward_single(x, adj, i)
        return self.mlp(x)

    def forward_single(self, x: Tensor, adj, cur_layer: int):
        out = self.convs[cur_layer](x, adj)
        if not self.reversible:
            out += self.skips[cur_layer](x)
        return out

    def training_step(self, batch, batch_idx: int):
        coords, feats, target = batch['coords'], batch['feats'], batch['labels']
        coords, feats = to_precision((coords, feats), self.trainer.precision)
        batch_idx = coords[:,0]
        edge_idx = torch_geometric.nn.pool.knn_graph(x=feats, k=16, batch=batch_idx,
                                                    loop=False, flow='source_to_target',
                                                    cosine=False, num_workers=1)
        y_hat = self(feats, edge_idx)
        train_loss = F.cross_entropy(y_hat, target)
        self.log('train_loss', train_loss, prog_bar=True, on_step=False,
                 on_epoch=True)
        preds = y_hat.argmax(dim=-1)
        return {'loss': train_loss, 'preds': preds, 'target': target}

    def validation_step(self, batch, batch_idx: int):
        coords, feats, target = batch['coords'], batch['feats'], batch['labels']
        coords, feats = to_precision((coords, feats), self.trainer.precision)
        batch_idx = coords[:,0]
        edge_idx = torch_geometric.nn.pool.knn_graph(x=feats, k=16, batch=batch_idx,
                                                    loop=False, flow='source_to_target',
                                                    cosine=False, num_workers=1)
        y_hat = self(feats, edge_idx)
        preds = y_hat.argmax(dim=-1)
        return {'loss': train_loss, 'preds': preds, 'target': target}

    def convert_sync_batchnorm(self):
        pass

    @staticmethod
    def add_argparse_args(parent_parser):
        parent_parser = BaseSegmentationModule.add_argparse_args(parent_parser)
        parser = parent_parser.add_argument_group("RevGNNSegModel")
        parser.add_argument("--hidden_channels", type=int, default=128)
        parser.add_argument("--heads", type=int, default=4)
        parser.add_argument("--num_layers", type=int, default=10)
        parser.add_argument("--group", type=int, default=2)
        parser.add_argument("--dropout", type=int, default=0.0)
        parser.add_argument("--norm", type=str, default='batch', choices=['batch', 'layer', 'instance', 'none'])
        parser.add_argument("--track_running_stats", type=str2bool, nargs='?', const=True, default=False)
        parser.add_argument("--reversible", type=str2bool, nargs='?', const=True, default=True)
        parser.add_argument("--model", type=str, default='rgat', choices=['rgat'])
        parser.add_argument("--mlp_channels", type=str, default='1,256,512')
        return parent_parser