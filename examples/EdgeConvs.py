from typing import Union, Tuple, Optional, Callable
from torch_geometric.typing import (PairTensor, OptPairTensor, Adj, Size, NoneType,
                                    OptTensor)

import torch
from torch import Tensor
import torch.nn as nn
from torch_geometric.nn.conv import MessagePassing

from typing import Union, Tuple, Callable, Optional
from torch_geometric.typing import PairTensor, Adj, OptTensor

from torch import Tensor
from torch_sparse import SparseTensor, set_diag

from torch_geometric.nn.dense.linear import Linear
from torch_geometric.utils import remove_self_loops, add_self_loops, softmax



class PointTransformerConv(MessagePassing):
    def __init__(self, in_channels: Union[int, Tuple[int, int]],
                 out_channels: int, pos_nn: Optional[Callable] = None,
                 attn_nn: Optional[Callable] = None,
                 add_self_loops: bool = False, **kwargs):
        kwargs.setdefault('aggr', 'mean')
        super().__init__(**kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.add_self_loops = add_self_loops

        if isinstance(in_channels, int):
            in_channels = (in_channels, in_channels)

        self.pos_nn = pos_nn
        if self.pos_nn is None:
            self.pos_nn = Linear(3, out_channels)

        self.attn_nn = attn_nn
        self.lin = Linear(in_channels[0], out_channels, bias=False)
        self.lin_src = Linear(in_channels[0], out_channels, bias=False)
        self.lin_dst = Linear(in_channels[1], out_channels, bias=False)

        self.reset_parameters()

    def reset_parameters(self):
        reset(self.pos_nn)
        if self.attn_nn is not None:
            reset(self.attn_nn)
        self.lin.reset_parameters()
        self.lin_src.reset_parameters()
        self.lin_dst.reset_parameters()


    def forward(
        self,
        x: Union[Tensor, PairTensor],
        pos: Union[Tensor, PairTensor],
        edge_index: Adj,
    ) -> Tensor:
        """"""
        if isinstance(x, Tensor):
            alpha = (self.lin_src(x), self.lin_dst(x))
            x: PairTensor = (self.lin(x), x)
        else:
            alpha = (self.lin_src(x[0]), self.lin_dst(x[1]))
            x = (self.lin(x[0]), x[1])

        if isinstance(pos, Tensor):
            pos: PairTensor = (pos, pos)

        if self.add_self_loops:
            if isinstance(edge_index, Tensor):
                edge_index, _ = remove_self_loops(edge_index)
                edge_index, _ = add_self_loops(
                    edge_index, num_nodes=min(pos[0].size(0), pos[1].size(0)))
            elif isinstance(edge_index, SparseTensor):
                edge_index = set_diag(edge_index)

        # propagate_type: (x: PairTensor, pos: PairTensor, alpha: PairTensor)
        out = self.propagate(edge_index, x=x, pos=pos, alpha=alpha, size=None)
        return out


    def message(self, x_j: Tensor, pos_i: Tensor, pos_j: Tensor,
                alpha_i: Tensor, alpha_j: Tensor, index: Tensor,
                ptr: OptTensor, size_i: Optional[int]) -> Tensor:

        # print(pos_i.shape, pos_j.shape)
        # print(pos_i, pos_j)
        delta = self.pos_nn(pos_i - pos_j)
        # print(delta)
        alpha = alpha_i - alpha_j + delta
        if self.attn_nn is not None:
            alpha = self.attn_nn(alpha)
        alpha = softmax(alpha, index, ptr, size_i)
        return alpha * (x_j + delta)

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels})')

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