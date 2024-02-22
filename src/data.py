import pandas as pd
import ase
from ase import Atom, Atoms
import ase.neighborlist
import numpy as np
import torch
import torch_geometric as tg
from torch_geometric.data import InMemoryDataset, Data

class GlassDynDataset(InMemoryDataset):
    def __init__(self, root, transform=None, pretransform=None):
        super(GlassDynDataset, self).__init__(root, transform, pretransform)
        self.data, self.slices = torch.load(self.processed_paths[0])
    
    @property
    def raw_file_names(self):
        return []
    @property
    def processed_file_names(self):
        return ['NaSi_time_seperate_800.dataset']

    def download(self):
        pass
    
    def process(self):
        default_dtype = torch.float32
        
        type_encoding = {}
        specie_am = []
        for ind, Z in enumerate([8, 11, 14]):
            specie = Atom(Z)
            type_encoding[specie.symbol] = ind
            specie_am.append(specie.mass)
        
        type_onehot = torch.eye(len(type_encoding))
        am_onehot = torch.tensor(specie_am)
        
        dataset = []
        
        pos_list, symbol, x, z, edge_src_list, edge_dst_list, edge_shift_list, edge_vec_list, edge_len_list, targets  = ([] for i in range(10))
        
        for i in range(1,401,1):
            obj = pd.read_pickle(f'{parent}/dataset_800/30Na_{i}.pickle')
            particle_type = obj["types"]
            type_dic = obj["type_dic"]
            symbols = list((pd.Series(particle_type.tolist())).map(type_dic))
            symbol.append(symbols)
            
            z.append(type_onehot[[type_encoding[specie] for specie in symbols]])
            x.append(am_onehot[[type_encoding[specie] for specie in symbols]])           
            pos = obj["start_positions"]
            particle_pos = torch.tensor(pos)
            pos_list.append(particle_pos)
            
            cell = obj["box"]
            lattice = torch.diag(torch.tensor(cell)).unsqueeze(0).float()
            edge_threshold = 5
            atoms = ase.Atoms(f'Si3000', positions=particle_pos, cell=cell, pbc=True)
            edge_src, edge_dst, edge_shift = ase.neighborlist.neighbor_list("ijS", a=atoms, cutoff=edge_threshold, self_interaction=False)
            edge_batch = particle_pos.new_zeros(particle_pos.shape[0], dtype=torch.long)[torch.from_numpy(edge_src)]
            edge_vec = (particle_pos[torch.from_numpy(edge_dst)]
                - particle_pos[torch.from_numpy(edge_src)]
                + torch.einsum('ni,nij->nj', torch.tensor(edge_shift, dtype=default_dtype), lattice[edge_batch]))
            edge_len = np.around(edge_vec.norm(dim=1).numpy(), decimals=2)
            
            edge_src_list.append(edge_src)
            edge_dst_list.append(edge_dst)
            edge_shift_list.append(edge_shift)
            edge_vec_list.append(edge_vec)
            edge_len_list.append(edge_len)
            
            trajectory_target_positions = obj["target_positions"]
            targets.append(np.mean([np.linalg.norm(t - pos, axis=-1) for t in trajectory_target_positions], axis=1))

        for ind in range(8):
            obj = pd.read_pickle(f'{parent}/dataset_800/30Na_1.pickle')
            times = obj["times"]
            time_features = torch.ones(len(symbols))*np.log10(times[ind])
            
            for i in range(400):
                dataset.append(tg.data.Data(
                    pos=pos_list[i].float(),
                    symbol=symbol[i],
                    x=x[i],
                    z=z[i],
                    time_features = time_features,
                    edge_index=torch.stack([torch.LongTensor(edge_src_list[i]), torch.LongTensor(edge_dst_list[i])], dim=0),
                    edge_shift=torch.tensor(edge_shift_list[i], dtype=default_dtype),
                    edge_vec=edge_vec_list[i].float(), 
                    edge_len=edge_len_list[i],
                    y=torch.tensor(targets[i][ind]).float()))
                
        data, slices = self.collate(dataset)
        torch.save((data, slices), self.processed_paths[0])