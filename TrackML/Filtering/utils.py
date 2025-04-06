import numpy as np
# pytorch 
import torch
import torch.nn as nn
import torch.utils.data as data
from torch import Tensor
# PyG
import torch_geometric
from torch_geometric.data import Data
from torch.utils.data.sampler import SubsetRandomSampler
from torch_geometric.utils import  to_networkx
from torch_geometric.loader import DataLoader
# TrackML
from TrackML.Models.utils import buildMLP
from TrackML.Embedding.dataset import PointCloudData
from TrackML.Embedding.base import EmbeddingBase
# networkx 
from networkx import connected_components

class PostEmbeddingGraph(data.Dataset): 
    
    # initialize the dataset class : 
    def __init__(
        self,dataset_path:str,
        detector_path:str,embd_model_path:str,
        min_nhits=3,margin:float=0.1, 
        max_num_neighbors:int=64
        )->None:
        '''
        dataset_path : path to the dataset with the events. 
        detector_path : path to the detector.csv file 
        min_nhits : upper bound on the number of  hits to keep. 
        margin : margin around which to create the radius graph 
        embd_model_path : path to the saved embedding model. 
        returns : For each event, returs the PyG graph data that 
            is constructed post embedding. 
        '''
        super().__init__()
        
        self.margin = margin 
        self.max_num_neighbors = max_num_neighbors
        
        # get the point cloud dataset: 
        self.dataset = PointCloudData(
            dataset_path=dataset_path, 
            detector_path=detector_path, 
            min_nhits=min_nhits
        )
        
        # save the embedding model 
        self.embd_model = EmbeddingBase.load_from_checkpoint(embd_model_path)
        self.embd_model.eval()
        
        
    def __len__(self)->int: 
        return len( self.dataset )
    
    def __getitem__(self, index)->Data:
        
        # get the raw node featuers and labels 
        node_feats , labels = self.dataset[index]
        # get the latent space feat of the nodes : 
        latent_feats = self.embd_model(node_feats)
        
        # get the radius graph edge index : 
        radius_graph_edge_index = torch_geometric.nn.pool.radius_graph(
            latent_feats, 
            r = self.margin , 
            loop = False , 
            max_num_neighbors=self.max_num_neighbors
        )
        
        # get the edges labels, 0 if they belong to the same track, 1 otherwise : 
        row , col = radius_graph_edge_index 
        edge_attr = (labels[row] != labels[col]).float().unsqueeze_(dim=1)
        
        # edge_purity ; 
        edge_purity = 1 - torch.sum( edge_attr )/radius_graph_edge_index.shape[1]
        
        # create the graph data structure : 
        graph_data = Data(
            x = node_feats, 
            edge_index=radius_graph_edge_index, 
            edge_attr=edge_attr, 
            y = labels , 
            edge_purity = edge_purity
        )
        
        return graph_data
    
    
def get_disconnected_components(data:Data):
    # Convert PyG graph to NetworkX graph
    G = to_networkx(data, to_undirected=True)
    
    # Get connected components
    component_list = list(connected_components(G))
    
    # Convert node indices back to PyTorch tensors
    components = [torch.tensor(list(component)) for component in component_list]
    
    return components

class FilteringModel(nn.Module): 
    
    def __init__(
        self, 
        in_features:int, 
        hidden_features:list, 
    )->None:
        
        super().__init__()
        
        # the acctual in featuers would be twice the the 
        # number of acctual node features since a edge feature 
        # is a concatination of two node featuers 
        
        in_features *= 2 
        # define the MLP layer : 
        self.MLP = buildMLP(
            insize=in_features, 
            outsize=1, 
            features=hidden_features,
            add_bnorm=True,
            add_activation=None
        )
        
    def forward(self,data:Data)->Tensor: 
        row , col = data.edge_index 
        edge_feats = torch.cat((data.x[row] , data.x[col]) , dim  = 1 )
        return self.MLP( edge_feats )
    
    
def train_test_split(
    dataset:PostEmbeddingGraph,
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