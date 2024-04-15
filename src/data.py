import pandas as pd
import ase
from ase import Atom, Atoms
import ase.neighborlist
import numpy as np
import torch
import torch_geometric as tg
from torch_geometric.data import InMemoryDataset, Data
from dscribe.descriptors import SOAP
from numba import jit
import math

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

def get_dist(list,cell):
    dim = [cell[0],cell[1],cell[2]]
    x_dif = np.abs(list[:,0][np.newaxis, :] - list[:,0][:, np.newaxis])
    y_dif = np.abs(list[:,1][np.newaxis, :] - list[:,1][:, np.newaxis])
    z_dif = np.abs(list[:,2][np.newaxis, :] - list[:,2][:, np.newaxis])
    x_dif = np.where(x_dif > 0.5 * dim[0], np.abs(x_dif - dim[0]), x_dif)
    y_dif = np.where(y_dif > 0.5 * dim[1], np.abs(y_dif - dim[1]), y_dif)
    z_dif = np.where(z_dif > 0.5 * dim[2], np.abs(z_dif - dim[2]), z_dif)
    i_i = np.sqrt(x_dif ** 2 + y_dif ** 2 + z_dif ** 2 )
    return i_i

def radial_feature(atoms, distances, center_index, neigh_index, cutoff):
    rrange = np.arange(1.6, cutoff + 0.2, 0.2)
    radial_features = np.zeros((rrange.shape[0],center_index.shape[0]))
    for ind_r, r in enumerate(rrange):
        for ind_c, c in enumerate(center_index):
            neighbors = np.where(distances[neigh_index,c]<cutoff)
            neighbor_dist = distances[neigh_index,c][neighbors]
            radial_features[ind_r,ind_c] = np.sum(np.exp(-((neighbor_dist - r)**2)/(0.2**2)))
    return radial_features

@jit
def G4_func(angles, distances, indicies):
    a = [14.633,14.633,14.633,14.633,2.554,2.554,2.554,2.554,1.648,1.648,1.204,1.204,1.204,1.204,0.933,0.933,0.933,0.933,0.695,0.695,0.695,0.695]
    b = [1,1,2,2,1,1,2,2,1,2,1,2,4,16,1,2,4,16,1,2,4,16]
    c = [-1,1,-1,1,-1,1,-1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]
    angle_func_sum = []
    for theta, angle_indicies in zip(angles, indicies):
        r_ij = distances[angle_indicies[0], angle_indicies[1]]
        r_ik = distances[angle_indicies[2], angle_indicies[1]]
        angle_func = []
        for a_i, b_i, c_i in zip(a,b,c):
            angle_func.append(math.exp(-(2*r_ij**2 + 2*r_ik**2 - 2*r_ij*r_ik*math.cos(theta))/a_i**2)*(1+b_i*math.cos(theta))**c_i)
        angle_func_sum.append(angle_func)
    return angle_func_sum


def angle_feature(atoms, distances, center_index, neigh_index,cutoff):
    angular_features = []
    for center in center_index:
        neighbors = np.where((distances[neigh_index,center]<cutoff) & (distances[neigh_index,center]>0))[0]
        if neighbors.shape[0] < 2:
            angular_features.append(np.full(22,0))
            continue

        upper_index = np.triu_indices(neighbors.shape[0], k=1)
        comb_1 = np.meshgrid(neighbors,neighbors)[0][upper_index]
        comb_2 = np.meshgrid(neighbors,neighbors)[1][upper_index]
        indicies = np.vstack((neigh_index[comb_1], np.full(len(comb_1),center), neigh_index[comb_2])).T
        angles = atoms.get_angles(indicies, mic=True)
        angular_features.append(np.sum(np.array(G4_func(angles, distances, indicies)),axis=0).flatten())
    return np.array(angular_features).T

def SOAP_func():
    species = ["Si", "Na", "O"]
    r_cut = 6.0
    n_max = 8
    l_max = 4

    # Setting up the SOAP descriptor
    soap = SOAP(
        species=species,
        periodic=True,
        r_cut=r_cut,
        n_max=n_max,
        l_max=l_max)
    return soap

def local_descriptor(descriptor_type, path):
    x = []
    if descriptor_type == "SOAP":
        SOAP_fn = SOAP_func()
        
    for i in range(1,401,1):
        print(i)
        obj = pd.read_pickle(f'{path}/30Na_{i}.pickle')
        particle_type = obj["types"]
        type_dic = obj["type_dic"]
        symbols = list((pd.Series(particle_type.tolist())).map(type_dic))
        pos = obj["start_positions"]
        cell = obj["box"]
    
        atoms = ase.Atoms(f'Si3000', positions=pos, cell=cell, pbc=True)
        atoms.set_chemical_symbols(symbols)
        Na_ind = np.where(np.array(symbols).flatten()=="Na")[0]
        Si_ind = np.where(np.array(symbols).flatten()=="Si")[0]
        O_ind = np.where(np.array(symbols).flatten()=="O")[0]
    
        if descriptor_type == "SOAP":
            x.append(SOAP_fn.create(atoms)[Na_ind])

        if descriptor_type == "ACSF":
            distances = get_dist(pos,cell)
            Na_O  = radial_feature(atoms, distances, Na_ind, O_ind, 6)
            Na_Na = radial_feature(atoms, distances, Na_ind, Na_ind, 6)
            Na_Si = radial_feature(atoms, distances, Na_ind, Si_ind, 6)
            O_Na_O = angle_feature(atoms, distances, Na_ind, O_ind, 6)
            Na_Na_Na = angle_feature(atoms, distances, Na_ind, Na_ind, 6)
            Si_Na_Si = angle_feature(atoms, distances, Na_ind, Si_ind, 6)
            x.append(np.vstack((Na_O,Na_Na,Na_Si,O_Na_O,Na_Na_Na,Si_Na_Si)))
    
    x = np.array(x)
    
    if descriptor_type == "ACSF":
        x = np.swapaxes(x,1,2)
        
    x_all = np.reshape(x,(x.shape[0]*x.shape[1],x.shape[2]))
    return x_all

def get_targets(path):
    y = []
    for i in range(1,401,1):
        obj = pd.read_pickle(f'{path}/30Na_{i}.pickle')
        particle_type = obj["types"]
        type_dic = obj["type_dic"]
        symbols = list((pd.Series(particle_type.tolist())).map(type_dic))
        Na_ind = np.where(np.array(symbols).flatten()=="Na")[0]
        pos = obj["start_positions"]
        trajectory_target_positions = obj["target_positions"]
        targets = np.mean([np.linalg.norm(t - pos, axis=-1) for t in trajectory_target_positions], axis=1)
        targets = targets[:,Na_ind]   
        y.append(targets)
    y = np.array(y)
    ys = np.swapaxes(y,1,2)
    y_all = np.reshape(ys,(ys.shape[0]*ys.shape[1],ys.shape[2]))
    return y_all