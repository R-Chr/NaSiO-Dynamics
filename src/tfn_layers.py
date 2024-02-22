import torch
from torch_scatter import scatter

import e3nn
from e3nn import o3
from e3nn import nn

from src.modules.irreps_tools import irreps2gate


class TensorProductConvLayer(torch.nn.Module):
    def __init__(
        self, 
        in_irreps,  
        out_irreps,
        sh_irreps,
        edge_feats_dim, 
        hidden_dim,
        avg_num_neighbors, 
        aggr="add",
    ):
        """Tensor Field Network GNN Layer
        
        Implements a Tensor Field Network equivariant GNN layer for higher-order tensors, using e3nn.
        Implementation adapted from: https://github.com/gcorso/DiffDock/

        Paper: Tensor Field Networks, Thomas, Smidt et al.

        Args:
            in_irreps: (e3nn.o3.Irreps) Input irreps dimensions
            out_irreps: (e3nn.o3.Irreps) Output irreps dimensions
            sh_irreps: (e3nn.o3.Irreps) Spherical harmonic irreps dimensions
            edge_feats_dim: (int) Edge feature dimensions
            hidden_dim: (int) Hidden dimension of MLP for computing tensor product weights
            aggr: (str) Message passing aggregator
        """
        super().__init__()
        self.in_irreps = in_irreps
        self.out_irreps = out_irreps
        self.sh_irreps = sh_irreps
        self.edge_feats_dim = edge_feats_dim
        self.aggr = aggr
        self.avg_num_neighbors = avg_num_neighbors

        # Tensor product over edges to construct messages
        self.tp = o3.FullyConnectedTensorProduct(in_irreps, sh_irreps, out_irreps, shared_weights=False)

        # MLP used to compute weights of tensor product
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(edge_feats_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, self.tp.weight_numel)
        )

        # Optional equivariant batch norm

    def forward(self, node_attr, edge_index, edge_attr, edge_feat):
        src, dst = edge_index
        # Compute messages 
        tp = self.tp(node_attr[dst], edge_attr, self.fc(edge_feat))
        # Aggregate messages
        out = scatter(tp, src, dim=0, reduce=self.aggr)
        return out / self.avg_num_neighbors
