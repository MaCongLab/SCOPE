from typing import Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Parameter

from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.nn.inits import glorot, zeros
from torch.nn.init import  kaiming_uniform_
from torch_geometric.typing import NoneType  # noqa
from torch_geometric.typing import (
    Adj,
    OptPairTensor,
    OptTensor,
    Size,
    SparseTensor,
    torch_sparse,
)
from torch_geometric.utils import (
    add_self_loops,
    is_torch_sparse_tensor,
    remove_self_loops,
    softmax,
)
from torch_geometric.utils.sparse import set_sparse_value


class MGAT(MessagePassing):
    def __init__(
        self,
        in_channels: Union[int, Tuple[int, int]],
        out_channels: int,
        heads: int = 1,
        concat: bool = True,
        negative_slope: float = 0.2,
        dropout: float = 0.0,
        add_self_loops: bool = True,
        edge_dim: Optional[int] = None,
        fill_value: Union[float, Tensor, str] = 'mean',
        bias: bool = True,
        **kwargs,
    ):
        kwargs.setdefault('aggr', 'add')
        super().__init__(node_dim=0, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.negative_slope = negative_slope
        self.dropout = dropout
        self.add_self_loops = add_self_loops
        self.edge_dim = edge_dim
        self.fill_value = fill_value

        if isinstance(in_channels, int):
            self.lin_src = Linear(in_channels, heads * out_channels,
                                  bias=False, weight_initializer='glorot')
            self.lin_dst = self.lin_src
        else:
            self.lin_src = Linear(in_channels[0], heads * out_channels, False,
                                  weight_initializer='glorot')
            self.lin_dst = Linear(in_channels[1], heads * out_channels, False,
                                  weight_initializer='glorot')


        if edge_dim is not None:
            self.lin_edge = Linear(edge_dim, heads * out_channels, bias=False,
                                   weight_initializer='glorot')
            self.att_edge = Linear(3*self.out_channels,self.out_channels,False,weight_initializer='glorot')
        else:
            self.lin_edge = None
            self.att_edge = Linear(2*self.out_channels,self.out_channels,False,weight_initializer='glorot')

        if bias and concat:
            self.bias = Parameter(torch.Tensor(heads * out_channels))
        elif bias and not concat:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        if self.concat:
            self.out_ln = Linear(self.heads*self.out_channels,self.heads*self.out_channels,False,weight_initializer='glorot')
        else:
            self.out_ln = Linear(self.out_channels, self.out_channels, False, weight_initializer='glorot')

        self.reset_parameters()

    def reset_parameters(self):
        super().reset_parameters()
        self.lin_src.reset_parameters()
        self.lin_dst.reset_parameters()
        self.out_ln.reset_parameters()
        if self.lin_edge is not None:
            self.lin_edge.reset_parameters()
        if self.att_edge is not None:
            self.att_edge.reset_parameters()
        zeros(self.bias)

    def forward(self, x: Union[Tensor, OptPairTensor], edge_index: Adj,
                edge_attr: OptTensor = None, size: Size = None,
                return_attention_weights=None):

        H, C = self.heads, self.out_channels

        x = self.lin_src(x).view(-1, H, C)


        if self.add_self_loops:
            num_nodes = x.size(0)
            edge_index, edge_attr = remove_self_loops(
                edge_index, edge_attr)
            edge_index, edge_attr = add_self_loops(
                edge_index, edge_attr, fill_value=self.fill_value,
                num_nodes=num_nodes)

        alpha,edge_feat = self.edge_updater(edge_index, alpha=x, edge_attr=edge_attr)

        out = self.propagate(edge_index, x=x, alpha=alpha, size=size,edge_feat=edge_feat)

        if self.concat:
            out = out.view(-1, self.heads * self.out_channels)
        else:
            out = out.mean(dim=1)

        out = self.out_ln(out)
        if self.bias is not None:
            out = out + self.bias

        edge_index, edge_feat = remove_self_loops(
            edge_index, edge_feat)
        edge_feat = torch.mean(edge_feat,dim=1)
        edge_feat = torch.squeeze(edge_feat,dim=1)

        if isinstance(return_attention_weights, bool):
            if isinstance(edge_index, Tensor):
                if is_torch_sparse_tensor(edge_index):
                    # TODO TorchScript requires to return a tuple
                    adj = set_sparse_value(edge_index, alpha)
                    return out, (adj, alpha)
                else:
                    return out, (edge_index, alpha)
            elif isinstance(edge_index, SparseTensor):
                return out, edge_index.set_value(alpha, layout='coo')
        else:
            return out,edge_feat

    def edge_update(self, alpha_j: Tensor, alpha_i: OptTensor,
                    edge_attr: OptTensor, index: Tensor, ptr: OptTensor,
                    size_i: Optional[int]):

        if edge_attr is not None and self.lin_edge is not None:
            edge_attr = self.lin_edge(edge_attr)
            edge_attr = edge_attr.view(-1, self.heads, self.out_channels)
            total_feat = torch.concat([alpha_i, alpha_j, edge_attr], dim=-1)
        else:
            total_feat = torch.concat([alpha_i, alpha_j], dim=-1)
        total_feat = self.att_edge(total_feat)
        alpha = softmax(total_feat, index, ptr, size_i)
        return alpha,edge_attr

    def message(self, x_j: Tensor, alpha: Tensor,edge_feat:Tensor) -> Tensor:
        if edge_feat is not None:
            return alpha * (x_j+edge_feat)
        else:
            return alpha * x_j

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels}, heads={self.heads})')
