import os
import numpy as np
# Pytorch 
import torch
import torch.nn as nn
from torch import Tensor
from torch.linalg import norm 
import torch.utils.data as data
# Pytorch Geometric: 
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torch_geometric.utils import  to_undirected, to_networkx
# Networkx
from networkx import connected_components
# Parent Module Imports:
from TrackML import Preprocessing
from TrackML.Models.utils import buildMLP 


    
### Data Set Definition: 
class PointCloudData(data.Dataset): 
    
    # initialize the dataset class : 
    def __init__(self,dataset_path:str,detector_path:str,min_nhits=3,max_r:float=800,drop_fake=True)->None:
        '''
        dataset_path : path to the dataset with the events. 
        eventids : list of eventid identifiers. 
        '''
        super().__init__()
        
        self.dataset_path = dataset_path
        self.min_nhits = min_nhits 
        
        self.detector = Preprocessing.load_detector_data(detector_path) 
        
        # get the list of event ids from the dataset folder : 
        eventids = [ code[:-9] for code in os.listdir(dataset_path) if code.endswith('-hits.csv') ]
        self.eventids = eventids
        
        self.max_r = max_r    
        self.drop_fake = drop_fake
        
    def __len__(self)->int: 
        return len( self.eventids )
    
    def __getitem__(self, index):
        node_data , labels =  (
            Preprocessing.process_event_data(
                train_path=self.dataset_path, 
                eventid=self.eventids[index], 
                detector=self.detector
            ), 
            Preprocessing.process_particle_labels(
                train_path=self.dataset_path, 
                eventid=self.eventids[index], 
                min_nhits=self.min_nhits 
            )
        )
        return Preprocessing.filter_hits(node_data,labels,self.max_r,self.drop_fake)
        
        
### def collate function for the dataloder : 
def collate_function(data_list): 
    return data_list[0] 


### Model Definition : 
class EmbeddingModel(nn.Module):
    
    def __init__(
        self, 
        in_features:int, 
        hidden_features:list, 
        out_features:int 
    )->None:
        
        super().__init__()
        
        # define the MLP layer : 
        self.MLP = buildMLP(
            insize=in_features, 
            outsize=out_features, 
            features=hidden_features,
            add_bnorm=True,
            add_activation=nn.BatchNorm1d(out_features)
        )
        
    def forward(self,x:Tensor)->Tensor: 
        return self.MLP(x)
    

### Custom Loss Function for the Model: 
def EmbeddingLossFunction(
        x:Tensor,
        labels:Tensor,
        radius_graph_edge_index:Tensor,
        margin:float=.01
    )->Tensor: 
    '''
    x  : model output in the latent space. shape = [nhits, out_feats]
    radius_graph_edge_index : graph formed with the radius ball algorithm. 
    positive_idx : pairs of indices of hits sharing the same 
        particle id. shape = [2,num_positive_pairs]. 
    negetive_idx : pairs of hits indices within the latent space 
        margin radius ball having different particle ids. 
    returns : the pair wise hinge loss. 
    '''
    
    radius_graph_edge_index = to_undirected( radius_graph_edge_index )
    
    # get the positive indices 
    positive_idx = Preprocessing.get_track_index_pairs(labels)
    
    # get the negetive indices pairs that lie within the margin ball : 
    row , col = radius_graph_edge_index

    # Create a mask to get those pairs which have different 
    # labels and avoid repeations by choosing row < col
    mask = ( row < col ) & ( labels[row] != labels[col] )
    negetive_idx = radius_graph_edge_index[:,mask]
    
    # get the positive row and col idx : 
    prow , pcol = positive_idx 
    # get the negetive row and col idx : 
    nrow , ncol = negetive_idx
    
    # delete variables not in use : 
    del positive_idx  , negetive_idx , radius_graph_edge_index
    
    # loss form the positive pairs 
    loss_plus = norm(
    x[prow , : ] - x[pcol, :] , 
    ord = 2 , 
    dim = -1 
    )

    # distance between negetive pairs that lie within the margin
    loss_minus = norm(
        x[ncol,:] - x[nrow,:], 
        ord = 2 , 
        dim = -1 
    )
    
    del pcol , ncol , prow , nrow , x 
    
    foo1 = (margin - loss_minus).sum()
    foo2 = loss_plus.sum()
    # final loss 
    loss = torch.log(foo1) + torch.log(1 + foo2/foo1 )
    
    del foo1 , foo2 , loss_plus, loss_minus
    return loss 

### get disconnected components from the graph: 
def get_disconnected_components(data:Data):
    # Convert PyG graph to NetworkX graph
    G = to_networkx(data, to_undirected=True)
    
    # Get connected components
    component_list = list(connected_components(G))
    
    # Convert node indices back to PyTorch tensors
    components = [torch.tensor(list(component)) for component in component_list]
    
    return components


### functions to split train/test/valid datasets : 
def train_test_split(
    dataset:PointCloudData,
    valid_size:float,
    test_size:float,
    num_works:int=4
):
    '''
    valid_size : amount of data to reserve for validation (normalized to 1 )
    test_size : amount of data to reserve for testing (normalized to 1 )
    Returns : train/validation/test data loders. 
    '''
    
    train_size=1-test_size-valid_size
    
    if not ( (train_size <= 1.) & (valid_size <= 1.) & (test_size <= 1. )) : 
        raise ValueError('Improper valid/train size encountered.')
    
    # total number of events : 
    num_events = len(dataset)
    
    # get shuffeled indices 
    indices = list(range(num_events))
    np.random.shuffle(indices)
    train_split = int(np.floor(train_size * num_events))
    valid_split = int(np.floor(valid_size * num_events))
    
    train_index, valid_index, test_index = indices[0:train_split], indices[train_split:train_split + valid_split], indices[train_split + valid_split:]
    
    # define samplers for obtaining training and validation batches
    train_sampler = SubsetRandomSampler(train_index)
    valid_sampler = SubsetRandomSampler(valid_index)
    test_sampler = SubsetRandomSampler(test_index)
    
    # define data loaders : 
    train_loder = DataLoader(
        dataset=dataset, 
        batch_size=1, 
        num_workers=num_works, 
        sampler = train_sampler,
        persistent_workers=True if num_works > 0 else False
    )
    
    valid_loder = DataLoader(
        dataset=dataset, 
        batch_size=1, 
        num_workers=num_works, 
        sampler = valid_sampler,
        persistent_workers=True if num_works > 0 else False
    )
    
    test_loder = DataLoader(
        dataset=dataset, 
        batch_size=1, 
        num_workers=num_works, 
        sampler = test_sampler,
        persistent_workers=True if num_works > 0 else False
    )
    
    # return data loders : 
    return train_loder,valid_loder,test_loder

### track putrity and particle purity metrics : 
def event_reconstruction_metrics( graph_data:Data , labels:Tensor )->tuple: 
    '''
        data : graph dataset to get the metrics for : 
        labels : particle_ids for each of the nodes (hits)
        returns : the trajecteory and particle purity of the graph dataset. 
    '''
    
    # get the disconnected compoenents (reconstructed tracks) form the graph steructure : 
    disconnected_components = get_disconnected_components(graph_data)
    
    
    ## 1. get the matched partices for each of the disconnecte graphs : 
    
    #  partices with the max occourence for each of the disconnected components : 
    max_track_particle = torch.tensor([ torch.mode(labels[component])[0] for component in disconnected_components])
    # print( max_track_particle )
    # frequency of the particle with the most occourence for each of the disconnected component : 
    max_track_particle_freq = torch.tensor([ torch.sum(labels[track] == particle) for particle,track in zip(max_track_particle,disconnected_components)])
    # print( max_track_particle_freq )
    # get the number of hits that belong to each reconstructed track : 
    num_hits_tracks = torch.tensor( [component.shape[0] for component in disconnected_components] )
    # total number of true hits left by the underlying max_track_particle : 
    max_track_particle_num_true_hits = torch.tensor([torch.sum(labels==particle) for particle in max_track_particle])
    
    return  max_track_particle_freq , num_hits_tracks , max_track_particle_num_true_hits