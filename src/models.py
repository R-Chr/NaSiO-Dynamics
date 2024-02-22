from typing import Callable, Optional, Union
import torch
from torch.nn import functional as F
import torch_geometric
from torch_geometric.nn import SchNet
import torch_scatter
from torch_scatter import scatter
from torch.nn import ReLU, SiLU, Softplus

from e3nn import o3

from src.modules.blocks import (
    EquivariantProductBasisBlock,
    RadialEmbeddingBlock,
)
from src.modules.irreps_tools import reshape_irreps
from src.egnn_layers import EGNNLayer
from src.tfn_layers import TensorProductConvLayer


class MACEModel(torch.nn.Module):
    def __init__(
        self,
        r_max=10.0,
        num_bessel=8,
        num_polynomial_cutoff=5,
        max_ell=2,
        correlation=3,
        num_layers=5,
        emb_dim=64,
        in_dim=1,
        out_dim=1,
        aggr="sum",
        residual=True,
        time_features=True,
        avg_num_neighbors=38,
    ):
        super().__init__()
        self.r_max = r_max
        self.emb_dim = emb_dim
        self.num_layers = num_layers
        self.residual = residual
        self.avg_num_neighbors=avg_num_neighbors
        
        # Embedding
        self.radial_embedding = RadialEmbeddingBlock(
            r_max=r_max,
            num_bessel=num_bessel,
            num_polynomial_cutoff=num_polynomial_cutoff,
        )
        sh_irreps = o3.Irreps.spherical_harmonics(max_ell)
        self.spherical_harmonics = o3.SphericalHarmonics(
            sh_irreps, normalize=True, normalization="component"
        )
        self.time_features = time_features
        
        # Embedding for initial node features
        self.emb_in = torch.nn.Linear(in_dim, emb_dim)

        self.convs = torch.nn.ModuleList()
        self.prods = torch.nn.ModuleList()
        self.reshapes = torch.nn.ModuleList()
        hidden_irreps = (sh_irreps * emb_dim).sort()[0].simplify()
        irrep_seq = [
            o3.Irreps(f'{emb_dim}x0e'),
            # o3.Irreps(f'{emb_dim}x0e + {emb_dim}x1o + {emb_dim}x2e'),
            # o3.Irreps(f'{emb_dim//2}x0e + {emb_dim//2}x0o + {emb_dim//2}x1e + {emb_dim//2}x1o + {emb_dim//2}x2e + {emb_dim//2}x2o'),
            hidden_irreps
        ]
        for i in range(num_layers):
            in_irreps = irrep_seq[min(i, len(irrep_seq) - 1)]
            out_irreps = irrep_seq[min(i + 1, len(irrep_seq) - 1)]
            conv = TensorProductConvLayer(
                in_irreps=in_irreps,
                out_irreps=out_irreps,
                sh_irreps=sh_irreps,
                edge_feats_dim=self.radial_embedding.out_dim,
                hidden_dim=emb_dim,
                aggr=aggr,
                avg_num_neighbors=avg_num_neighbors
            )
            self.convs.append(conv)
            self.reshapes.append(reshape_irreps(out_irreps))
            prod = EquivariantProductBasisBlock(
                node_feats_irreps=out_irreps,
                target_irreps=out_irreps,
                correlation=correlation,
                element_dependent=False,
                num_elements=in_dim,
                use_sc=residual,
            )
            self.prods.append(prod)

        # Predictor MLP
        self.pred = torch.nn.Sequential(
            torch.nn.Linear(emb_dim, emb_dim),
            torch.nn.Softplus(),
            torch.nn.Linear(emb_dim, out_dim)
        )

    
    def forward(self, batch):
        input_features = batch.z
        
        if self.time_features:
            input_features = torch.cat((input_features, batch.time_features.view(-1, 1)), dim=1).float()
        
        h = self.emb_in(input_features)  # (n,) -> (n, d)

        # Edge features
        vectors = batch.edge_vec  # [n_edges, 3]
        lengths = torch.linalg.norm(vectors, dim=-1, keepdim=True)  # [n_edges, 1]
        edge_attrs = self.spherical_harmonics(vectors)
        edge_feats = self.radial_embedding(lengths)
        
        for conv, reshape, prod in zip(self.convs, self.reshapes, self.prods):
            # Message passing layer
            h_update = conv(h, batch.edge_index, edge_attrs, edge_feats)
            # Update node features
            sc = F.pad(h, (0, h_update.shape[-1] - h.shape[-1]))
            h = prod(reshape(h_update), sc, None)

        h = h[:,:self.emb_dim]
        h = self.pred(h)
        
        return h  # (batch_size, out_dim)

class TFNModel(torch.nn.Module):
    def __init__(
        self,
        r_max=10.0,
        num_bessel=8,
        num_polynomial_cutoff=5,
        max_ell=2,
        num_layers=5,
        emb_dim=64,
        in_dim=1,
        out_dim=1,
        avg_num_neighbors=38,
        aggr="sum",
        residual=True,
        time_features=True,
    ):
        super().__init__()
        self.r_max = r_max
        self.emb_dim = emb_dim
        self.num_layers = num_layers
        self.residual = residual
        self.time_features = time_features
        self.avg_num_neighbors=avg_num_neighbors
        
        # Embedding
        self.radial_embedding = RadialEmbeddingBlock(
            r_max=r_max,
            num_bessel=num_bessel,
            num_polynomial_cutoff=num_polynomial_cutoff,
        )
        sh_irreps = o3.Irreps.spherical_harmonics(max_ell)
        self.spherical_harmonics = o3.SphericalHarmonics(
            sh_irreps, normalize=True, normalization="component"
        )

        # Embedding for initial node features
        self.emb_in = torch.nn.Linear(in_dim, emb_dim)
        
        self.convs = torch.nn.ModuleList()
        hidden_irreps = (sh_irreps * emb_dim).sort()[0].simplify()
        irrep_seq = [
            o3.Irreps(f'{emb_dim}x0e'),
            # o3.Irreps(f'{emb_dim}x0e + {emb_dim}x1o + {emb_dim}x2e'),
            # o3.Irreps(f'{emb_dim//2}x0e + {emb_dim//2}x0o + {emb_dim//2}x1e + {emb_dim//2}x1o + {emb_dim//2}x2e + {emb_dim//2}x2o'),
            hidden_irreps
        ]
        for i in range(num_layers):
            in_irreps = irrep_seq[min(i, len(irrep_seq) - 1)]
            out_irreps = irrep_seq[min(i + 1, len(irrep_seq) - 1)]
            conv = TensorProductConvLayer(
                in_irreps=in_irreps,
                out_irreps=out_irreps,
                sh_irreps=sh_irreps,
                edge_feats_dim=self.radial_embedding.out_dim,
                hidden_dim=emb_dim,
                aggr=aggr,
                avg_num_neighbors=avg_num_neighbors
            )
            self.convs.append(conv)

        # Predictor MLP
        self.pred = torch.nn.Sequential(
            torch.nn.Linear(emb_dim, emb_dim),
            torch.nn.Softplus(),
            torch.nn.Linear(emb_dim, out_dim)
        )
    
    def forward(self, batch):
        input_features = batch.z
        
        if self.time_features:
            input_features = torch.cat((input_features, batch.time_features.view(-1, 1)), dim=1).float()
        
        h = self.emb_in(input_features)  # (n,) -> (n, d)
        
        # Edge features
        vectors = batch.edge_vec  # [n_edges, 3]
        lengths = torch.linalg.norm(vectors, dim=-1, keepdim=True)  # [n_edges, 1]
        edge_attrs = self.spherical_harmonics(vectors)
        edge_feats = self.radial_embedding(lengths)
        
        for conv in self.convs:
            # Message passing layer
            h_update = conv(h, batch.edge_index, edge_attrs, edge_feats)

            # Update node features
            h = h_update + F.pad(h, (0, h_update.shape[-1] - h.shape[-1])) if self.residual else h_update

        # Select only scalars for prediction
        h = h[:,:self.emb_dim]
        h = self.pred(h)
        
        return h  # (batch_size, out_dim)

class EGNNModel(torch.nn.Module):
    def __init__(
        self,
        num_layers=5,
        emb_dim=128,
        in_dim=1,
        out_dim=1,
        activation="relu",
        norm="layer",
        aggr="sum",
        residual=True,
        time_features=True,
    ):
        """E(n) Equivariant GNN model 
        
        Args:
            num_layers: (int) - number of message passing layers
            emb_dim: (int) - hidden dimension
            in_dim: (int) - initial node feature dimension
            out_dim: (int) - output number of classes
            activation: (str) - non-linearity within MLPs (swish/relu)
            norm: (str) - normalisation layer (layer/batch)
            aggr: (str) - aggregation function `\oplus` (sum/mean/max)
            residual: (bool) - whether to use residual connections
        """
        super().__init__()
        self.activation = {"swish": SiLU(), "relu": ReLU(), "softplus": Softplus()}[activation]
        self.time_features = time_features
        
        # Embedding for initial node features
        self.emb_in = torch.nn.Linear(in_dim, emb_dim)
        
        # Stack of GNN layers
        self.convs = torch.nn.ModuleList()
        for layer in range(num_layers):
            self.convs.append(EGNNLayer(emb_dim, activation, norm, aggr))

        # Predictor MLP
        self.pred = torch.nn.Sequential(
            torch.nn.Linear(emb_dim, emb_dim),
            self.activation,
            torch.nn.Linear(emb_dim, out_dim)
        )
        self.residual = residual

    def forward(self, batch):
        input_features = batch.z
        
        if self.time_features:
            input_features = torch.cat((input_features, batch.time_features.view(-1, 1)), dim=1).float()
        
        h = self.emb_in(input_features)  # (n,) -> (n, d)
        
        pos = batch.pos  # (n, 3)

        for conv in self.convs:
            # Message passing layer
            h_update, pos_update = conv(h, pos, batch.edge_index)

            # Update node features (n, d) -> (n, d)
            h = h + h_update if self.residual else h_update 

            # Update node coordinates (no residual) (n, 3) -> (n, 3)
            pos = pos_update

        return self.pred(h)  # (batch_size, out_dim)


class SchNetModel(SchNet):
    def __init__(
        self, 
        hidden_channels: int = 128, 
        in_dim: int = 1,
        out_dim: int = 1, 
        num_filters: int = 128, 
        num_layers: int = 6,
        num_gaussians: int = 50, 
        cutoff: float = 10, 
        max_num_neighbors: int = 32, 
        readout: str = 'add', 
        dipole: bool = False,
        mean: Optional[float] = None, 
        std: Optional[float] = None, 
        atomref: Optional[torch.Tensor] = None,
        time_features=True,
    ):
        super().__init__(hidden_channels, num_filters, num_layers, num_gaussians, cutoff, max_num_neighbors, readout, dipole, mean, std, atomref)

        # Overwrite atom embedding and final predictor
        self.lin2 = torch.nn.Linear(hidden_channels // 2, out_dim)
        self.emb_in = torch.nn.Linear(in_dim, hidden_channels)
        self.time_features = time_features
        
    def forward(self, batch):
        input_features = batch.z
        
        if self.time_features:
            input_features = torch.cat((input_features, batch.time_features.view(-1, 1)), dim=1).float()
        
        h = self.emb_in(input_features)  # (n,) -> (n, d)
        
        row, col = batch.edge_index
        vectors = batch.edge_vec  # [n_edges, 3]
        edge_weight = torch.linalg.norm(vectors, dim=-1, keepdim=True)  # [n_edges, 1]
        edge_attr = self.distance_expansion(edge_weight)

        for interaction in self.interactions:
            h = h + interaction(h, batch.edge_index, edge_weight, edge_attr)

        h = self.lin1(h)
        h = self.act(h)
        h = self.lin2(h)
        return h