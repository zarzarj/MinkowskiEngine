from typing import Optional, Callable, Union, Any

import torch
from torch import Tensor
from torch_sparse import SparseTensor, set_diag
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import remove_self_loops, add_self_loops
from torch_geometric.typing import OptTensor, PairOptTensor, PairTensor, Adj


def reset(value: Any):
    if hasattr(value, 'reset_parameters'):
        value.reset_parameters()
    else:
        for child in value.children() if hasattr(value, 'children') else []:
            reset(child)

class PointConv_NoCoords(MessagePassing):

    def __init__(self, local_nn: Optional[Callable] = None,
                 global_nn: Optional[Callable] = None,
                 add_self_loops: bool = True, **kwargs):
        kwargs.setdefault('aggr', 'max')
        super().__init__(**kwargs)

        self.local_nn = local_nn
        self.global_nn = global_nn
        self.add_self_loops = add_self_loops

        self.reset_parameters()

    def reset_parameters(self):
        reset(self.local_nn)
        reset(self.global_nn)


    def forward(self, x: Union[OptTensor, PairOptTensor],
                pos: Union[Tensor, PairTensor], edge_index: Adj) -> Tensor:
        """"""
        if not isinstance(x, tuple):
            x: PairOptTensor = (x, None)

        if isinstance(pos, Tensor):
            pos: PairTensor = (pos, pos)

        if self.add_self_loops:
            if isinstance(edge_index, Tensor):
                edge_index, _ = remove_self_loops(edge_index)
                edge_index, _ = add_self_loops(
                    edge_index, num_nodes=min(pos[0].size(0), pos[1].size(0)))
            elif isinstance(edge_index, SparseTensor):
                edge_index = set_diag(edge_index)

        # propagate_type: (x: PairOptTensor, pos: PairTensor)
        out = self.propagate(edge_index, x=x, pos=pos, size=None)

        if self.global_nn is not None:
            out = self.global_nn(out)

        return out


    def message(self, x_j: Optional[Tensor], pos_i: Tensor,
                pos_j: Tensor) -> Tensor:
        msg = x_j
        if self.local_nn is not None:
            msg = self.local_nn(msg)
        return msg

    def __repr__(self):
        return '{}(local_nn={}, global_nn={})'.format(self.__class__.__name__,
                                                      self.local_nn,
                                                      self.global_nn)