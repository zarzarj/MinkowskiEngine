from typing import Union, Tuple, Optional, Callable
from torch_geometric.typing import (PairTensor, OptPairTensor, Adj, Size, NoneType,
                                    OptTensor)

import torch
from torch import Tensor
import torch.nn as nn
from torch_geometric.nn.conv import MessagePassing


def reset(nn):
    def _reset(item):
        if hasattr(item, 'reset_parameters'):
            item.reset_parameters()

    if nn is not None:
        if hasattr(nn, 'children') and len(list(nn.children())) > 0:
            for item in nn.children():
                _reset(item)
        else:
            _reset(nn)

class DenseMRConv(torch.nn.Module):
    def __init__(self, nn: Callable, aggr: str = 'max', **kwargs):
        super(DenseMRConv, self).__init__(**kwargs)
        self.nn = nn
        self.aggr = aggr
        self.reset_parameters()

    def reset_parameters(self):
        reset(self.nn)

    def forward(self, x: Tensor, edge_index: Adj) -> Tensor:
        """"""
        x_i = x.unsqueeze(1).repeat(1, edge_index.shape[-1], 1)
        x_j = x[edge_index.long(), :]
        # print(x_i.shape, x_j.shape)
        x = torch.cat([x_i, x_j - x_i], dim=-1)
        # print(x.shape)
        if self.aggr == 'max':
            x = torch.max(x, dim = 1)[0]
        elif self.aggr == 'sum':
            x = torch.sum(x, dim = 1)
        elif self.aggr == 'mean':
            x = torch.mean(x, dim = 1)
        x = self.nn(x)
        return x

    def __repr__(self):
        return '{}(nn={})'.format(self.__class__.__name__, self.nn)

class EffSparseEdgeConv(nn.Module):
    def __init__(self, in_channels, out_channels, bias: bool = True, aggr: str = 'add', **kwargs):
        super(EffSparseEdgeConv, self).__init__(**kwargs)
        self.in_channels = in_channels
        if aggr == 'max':
            aggr = 'add'
        self.aggr = aggr
        self.lin1 = nn.Linear(in_channels, out_channels, bias=bias)
        self.lin2 = nn.Linear(in_channels, out_channels, bias=False)
        self.reset_parameters()

    def reset_parameters(self):
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self, x: Tensor, edge_index: Adj) -> Tensor:
        """"""
        # print(x.shape, x)
        # print(x.dtype)
        # edge_index = edge_index.type_as(x)
        edge_index = edge_index.to_torch_sparse_coo_tensor().to(x.dtype)
        num_neighbors = torch.sparse.mm(edge_index, torch.ones((x.shape[0], 1)).type_as(x))

        x_1 = self.lin1(x)
        x_2 = self.lin2(x)
        if self.aggr == 'add':
            x_i = (x_1 - x_2) * num_neighbors
            x_j = torch.sparse.mm(edge_index, x_2)
        elif self.aggr == 'mean':
            x_i = (x_1 - x_2)
            x_j = torch.sparse.mm(edge_index, x_2) / num_neighbors
        return x_i + x_j

    def __repr__(self):
        return '{}({})'.format(self.__class__.__name__, self.lin1)

class EffSparseEdgeOnlyConv(nn.Module):
    def __init__(self, in_channels, out_channels, bias: bool = True, aggr: str = 'add', **kwargs):
        super(EffSparseEdgeOnlyConv, self).__init__(**kwargs)
        self.in_channels = in_channels
        self.aggr = aggr
        self.lin2 = nn.Linear(in_channels, out_channels, bias=bias)
        self.reset_parameters()

    def reset_parameters(self):
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self, x: Tensor, edge_index: Adj) -> Tensor:
        """"""
        # print(x.shape, x)
        # print(x.dtype)
        # edge_index = edge_index.type_as(x)
        edge_index = edge_index.to_torch_sparse_coo_tensor().to(x.dtype)
        num_neighbors = torch.sparse.mm(edge_index, torch.ones((x.shape[0], 1)).type_as(x))

        x_2 = self.lin2(x)
        if self.aggr == 'add':
            x_i = - x_2 * num_neighbors
            x_j = torch.sparse.mm(edge_index, x_2)
        elif self.aggr == 'mean':
            x_i = - x_2
            x_j = torch.sparse.mm(edge_index, x_2) / num_neighbors
        return x_i + x_j

    def __repr__(self):
        return '{}({})'.format(self.__class__.__name__, self.lin1)

class EdgeOnlyConv(MessagePassing):
    def __init__(self, nn: Callable, aggr: str = 'max', **kwargs):
        super(EdgeOnlyConv, self).__init__(aggr=aggr, **kwargs)
        self.nn = nn
        self.reset_parameters()

    def reset_parameters(self):
        reset(self.nn)

    def forward(self, x: Union[Tensor, PairTensor], edge_index: Adj) -> Tensor:
        """"""
        if isinstance(x, Tensor):
            x: PairTensor = (x, x)
        # propagate_type: (x: PairTensor)
        # print(x.shape)
        return self.propagate(edge_index, x=x, size=None)


    def message(self, x_i: Tensor, x_j: Tensor) -> Tensor:
        return self.nn(x_j - x_i)

    def __repr__(self):
        return '{}(nn={})'.format(self.__class__.__name__, self.nn)


class DenseEdgeOnlyConv(torch.nn.Module):
    def __init__(self, nn: Callable, aggr: str = 'mean', **kwargs):
        super(DenseEdgeOnlyConv, self).__init__(**kwargs)
        self.nn = nn
        self.reset_parameters()
        self.aggr = aggr

    def reset_parameters(self):
        reset(self.nn)

    def forward(self, x: Tensor, edge_index: Adj) -> Tensor:
        """"""
        # print(x, x.shape)

        x_j = x[edge_index.long(), :].contiguous().reshape(-1,x.shape[-1])
        x_i = x.unsqueeze(1).repeat(1, edge_index.shape[-1], 1).contiguous().reshape(-1,x.shape[-1])

        x = self.nn(x_j - x_i).contiguous().reshape(edge_index.shape[0], edge_index.shape[1], -1)
        if self.aggr == 'max':
            x = torch.max(x, dim = 1)[0]
        elif self.aggr == 'sum':
            x = torch.sum(x, dim = 1)
        elif self.aggr == 'mean':
            x = torch.mean(x, dim = 1)
        return x

    def __repr__(self):
        return '{}(nn={})'.format(self.__class__.__name__, self.nn)

class DenseEdgeConv(torch.nn.Module):
    def __init__(self, nn: Callable, aggr: str = 'mean', **kwargs):
        super(DenseEdgeConv, self).__init__(**kwargs)
        self.nn = nn
        self.reset_parameters()
        self.aggr = aggr

    def reset_parameters(self):
        reset(self.nn)

    def forward(self, x: Tensor, edge_index: Adj) -> Tensor:
        """"""
        # print(x, x.shape)
        x_i = x.unsqueeze(1).repeat(1, edge_index.shape[-1], 1)
        x_i = x_i.contiguous().reshape(-1, x.shape[-1])
        x_j = x[edge_index.long(), :]
        x_j = x_j.contiguous().reshape(-1, x.shape[-1])

        x = torch.cat([x_i, x_j - x_i], dim=-1)
        # print(x_i.shape, x_j.shape, x.shape)
        x = self.nn(x).contiguous().reshape(edge_index.shape[0], edge_index.shape[1], -1)
        if self.aggr == 'max':
            x = torch.max(x, dim = 1)[0]
        elif self.aggr == 'sum':
            x = torch.sum(x, dim = 1)
        elif self.aggr == 'mean':
            x = torch.mean(x, dim = 1)
        # print(x.shape)
        return x

    def __repr__(self):
        return '{}(nn={})'.format(self.__class__.__name__, self.nn)

class DenseEdgeConvTriplets(torch.nn.Module):
    def __init__(self, nn: Callable, aggr: str = 'mean', **kwargs):
        super(DenseEdgeConvTriplets, self).__init__(**kwargs)
        self.nn = nn
        self.reset_parameters()
        self.aggr = aggr

    def reset_parameters(self):
        reset(self.nn)

    def forward(self, x: Tensor, edge_index: Adj) -> Tensor:
        """"""
        # print(x, x.shape)
        x_i = x.unsqueeze(1).repeat(1, edge_index.shape[-1], 1)
        x_i = x_i.unsqueeze(1).repeat(1, edge_index.shape[-1], 1, 1)
        x_i = x_i.contiguous().reshape(-1, x.shape[-1])
        x_j = x[edge_index.long(), :]
        x_j = x_j.unsqueeze(1).repeat(1, edge_index.shape[-1], 1, 1)
        x_j = x_j.contiguous().reshape(-1, x.shape[-1])
        x_k = x[edge_index.long(), :]
        x_k = x_k.unsqueeze(1).repeat_interleave(edge_index.shape[-1], dim=1)
        x_k = x_k.contiguous().reshape(-1, x.shape[-1])

        x = torch.cat([x_i, x_j - x_i, x_k - x_i], dim=-1)
        x = self.nn(x).contiguous().reshape(edge_index.shape[0], edge_index.shape[1] * edge_index.shape[1], -1)
        if self.aggr == 'max':
            x = torch.max(x, dim = 1)[0]
        elif self.aggr == 'sum':
            x = torch.sum(x, dim = 1)
        elif self.aggr == 'mean':
            x = torch.mean(x, dim = 1)
        # print(x.shape)
        return x

    def __repr__(self):
        return '{}(nn={})'.format(self.__class__.__name__, self.nn)

class EdgeConvZ(MessagePassing):
    def __init__(self, nn: Callable, aggr: str = 'max', **kwargs):
        super(EdgeConvZ, self).__init__(aggr=aggr, **kwargs)
        self.nn = nn
        self.reset_parameters()

    def reset_parameters(self):
        reset(self.nn)


    def forward(self, x: Union[Tensor, PairTensor], edge_index: Adj) -> Tensor:
        """"""
        if isinstance(x, Tensor):
            x: PairTensor = (x, x)
        # propagate_type: (x: PairTensor)
        return self.propagate(edge_index, x=x, size=None)


    def message(self, x_i: Tensor, x_j: Tensor) -> Tensor:
        
        return self.nn(torch.cat([x_i[:,2].unsqueeze(-1), x_j - x_i], dim=-1))

    def __repr__(self):
        return '{}(nn={})'.format(self.__class__.__name__, self.nn)

class Embedder:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.create_embedding_fn()

    def create_embedding_fn(self):
        embed_fns = []
        d = self.kwargs['input_dims']
        out_dim = 0
        if self.kwargs['include_input']:
            embed_fns.append(lambda x : x)
            out_dim += d

        max_freq = self.kwargs['max_freq_log2']
        N_freqs = self.kwargs['num_freqs']

        if self.kwargs['log_sampling']:
            freq_bands = 2.**torch.linspace(0., max_freq, steps=N_freqs)
        else:
            freq_bands = torch.linspace(2.**0., 2.**max_freq, steps=N_freqs)

        for freq in freq_bands:
            for p_fn in self.kwargs['periodic_fns']:
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq : p_fn(x * freq))
                out_dim += d

        self.embed_fns = embed_fns
        self.out_dim = out_dim

    def embed(self, inputs):
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)


def get_embedder(multires=10):
    if multires == -1:
        return nn.Identity(), 3

    embed_kwargs = {
                'include_input' : True,
                'input_dims' : 3,
                'max_freq_log2' : multires-1,
                'num_freqs' : multires,
                'log_sampling' : True,
                'periodic_fns' : [torch.sin, torch.cos],
    }

    embedder_obj = Embedder(**embed_kwargs)
    embed = lambda x, eo=embedder_obj : eo.embed(x)
    return embed, embedder_obj.out_dim

class EdgeOnlyConvPos(MessagePassing):
    def __init__(self, nn: Callable, aggr: str = 'max', **kwargs):
        super(EdgeOnlyConvPos, self).__init__(aggr=aggr, **kwargs)
        self.nn = nn
        self.reset_parameters()
        self.input_pos_embedder, self.pos_features = get_embedder(10)

    def reset_parameters(self):
        reset(self.nn)


    def forward(self, x: Union[Tensor, PairTensor], edge_index: Adj) -> Tensor:
        """"""
        if isinstance(x, Tensor):
            x: PairTensor = (x, x)
        # propagate_type: (x: PairTensor)
        return self.propagate(edge_index, x=x, size=None)


    def message(self, x_i: Tensor, x_j: Tensor) -> Tensor:
        x_pos = self.input_pos_embedder(x_j[:,:3] - x_i[:,:3])
        x_edge = torch.cat([x_pos, x_j[:,3:] - x_i[:,3:]], dim=-1)
        return self.nn(x_edge)

    def __repr__(self):
        return '{}(nn={})'.format(self.__class__.__name__, self.nn)
