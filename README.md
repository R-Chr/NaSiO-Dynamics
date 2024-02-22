# Predicting dynamics in Sodium Silicate glasses

Repo build upon GNN implementations from Geometric GNN Dojo (https://github.com/chaitjo/geometric-gnn-dojo) by Joshi et al. [PDF](https://arxiv.org/pdf/2301.09308.pdf)

## Architectures

The `/src` directory provides implementations geometric GNN architectures:
- Invariant GNNs: [SchNet](https://arxiv.org/abs/1706.08566)
- Equivariant GNNs using spherical tensors: [Tensor Field Network](https://arxiv.org/abs/1802.08219), [MACE](http://arxiv.org/abs/2206.07697)

## Installation

```bash
# Create new conda environment
conda create --prefix ./env python=3.8
conda activate ./env

# Install PyTorch (Check CUDA version for GPU!)
#
# Option 1: CPU
conda install pytorch==1.12.0 -c pytorch
#
# Option 2: GPU, CUDA 11.3
# conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch

# Install dependencies
conda install matplotlib pandas networkx
conda install jupyterlab -c conda-forge
pip install e3nn==0.4.4 ipdb ase

# Install PyG (Check CPU/GPU/MacOS)
#
# Option 1: CPU, MacOS
pip install torch-scatter torch-sparse torch-cluster torch-spline-conv -f https://data.pyg.org/whl/torch-1.12.0+cpu.html 
pip install torch-geometric
#
# Option 2: GPU, CUDA 11.3
# pip install torch-scatter torch-sparse torch-cluster torch-spline-conv -f https://data.pyg.org/whl/torch-1.12.1+cu113.html
# pip install torch-geometric
#
# Option 3: CPU/GPU, but may not work on MacOS
# conda install pyg -c pyg
```


## Directory Structure and Usage

```
.
├── README.md
| 
└── src                                 # Geometric GNN models library
    ├── models.py                       # Models built using layers
    ├── tfn_layers.py                   # Layers for Tensor Field Networks
    ├── modules                         # Layers for MACE
    └── utils                           # Helper functions for training, plotting, etc.
```

## Contact

Rasmus Christensen (rasmusc@bio.aau.dk)

