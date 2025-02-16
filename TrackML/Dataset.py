# import libraries : 
import gc 
import os 
import random 
import numpy as np 
import pandas as pd 
from typing import Optional, Tuple 

# set random seed : 
np.random.seed( 41 )
random.seed( 41 )

import torch
import psutil
import torch.nn as nn 
from torch import cdist
from torch import Tensor 
import torch.nn.functional as F 
import torch.utils.data as data 
from torch.utils.data.sampler import SubsetRandomSampler

import torch_geometric
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.utils import  remove_self_loops , scatter 


# check if CPU is available for training : 

device = 'gpu'
if torch.cuda.is_available(): 
    device = 'cuda'
elif torch.mps.is_available(): 
    device = 'mps'

device = torch.device( device )


class EventData(data.Dataset): 
    
    # initialaize the event dataset 
    def __init__(self,path:str,device=device,threshold_dist:float=20)->None:
        '''
        Inputs : 
            path: path to the folder where the csv file was contained.  
        '''
        super(EventData,self).__init__()
        self.events = [code[:-9] for code in os.listdir(path) if code.endswith('-hits.csv')]
        self.num_events = len(self.events)
        self.threshold_dist  = threshold_dist
        self.path = path
        self.device = device  
        
    # function returns graph type represntation of the event dataset 
    def GraphData(self,idx:int) -> Data :
        eventid = self.events[idx] 
        
        # read the required csv files : 
        hits = pd.read_csv(self.path+eventid+'-hits.csv')
        truth = pd.read_csv(self.path+eventid+'-truth.csv')
        cells = pd.read_csv(self.path+eventid+'-cells.csv')
        particles = pd.read_csv( self.path+eventid+'-particles.csv')
        particle_ids  = truth.particle_id   
        # get charges corrsponding to the hits : 
        charge = Tensor([ particles.loc[ particles.particle_id.index[ particles.particle_id == ids  ] , 'q' ].tolist()[0] if ids != 0 else 0  for ids in particle_ids ])
        del particles , particle_ids 
        gc.collect()
        # find the charges left on the hit ( q = +- 1 )
        
        
        # total number of hits : these form the NODES of our graph. 
        nhits = hits.shape[0] 
        # x , y , z spatial featuers of the hits:  
        hits_spatial = hits.to_numpy()[: , 1:4 ]
        # Add a new feature vector : the number of cells that detect the hit : 
        node_fets = np.concatenate(
            (
                hits_spatial ,
                cells.hit_id.value_counts().get( hits.hit_id , 0 ).to_numpy().reshape((-1,1))
            ), 
            axis = 1 
        )
        del cells 
        gc.collect()
        # id's related to the hits 
        # this will help to initialize the graph structure : 
        hit_ids = Tensor(hits.hit_id.to_numpy( )).int()
        volume_id = Tensor(hits.volume_id.to_numpy( ))
        layer_id = Tensor(hits.layer_id.to_numpy( ))
        
        # get the particle true hit position and momentum, we add this to the node feat matrix : 
        node_fets = np.concatenate(
            (
                node_fets , 
                truth[['tx' , 'ty' , 'tz'  ]].to_numpy() - hits_spatial , 
                truth[['tpx' , 'tpy' ,'tpz']].to_numpy()
            ), 
            axis = 1 
        )
        node_fets = Tensor( node_fets )
        hits_spatial = Tensor( hits_spatial )
        
        # here we create edge_index's for the graph skeleton : 
        batch_size = 10000  # Process 10K nodes at a time
        edges = []
        
        
        for i in range(0, nhits, batch_size):
            # set distance threshold : 
            mask = cdist( hits_spatial[i : i + batch_size , : ] , hits_spatial , p = 2 ) < self.threshold_dist
            # mask2 ensures hits are either conect to another hit iff the volume_id of the dst > volume if of src or 
            # layer id of dst > layer id of src in case they have the same volume id 
            mask2 = volume_id.unsqueeze(0) -  volume_id[i:i+batch_size].unsqueeze(1) >= 0  
            mask2 = mask2 | ((volume_id.unsqueeze(0) == volume_id[i:i+batch_size].unsqueeze(1) ) & (layer_id.unsqueeze(0) - layer_id[i:i+batch_size].unsqueeze(1) >= 0 ))
            mask = mask & ( charge.unsqueeze(0) - charge[i:i+batch_size].unsqueeze(1) == 0 )
            # ensure both conditions are satisfied 
            mask = mask & mask2 
            del mask2 
            gc.collect()
            src, dst = torch.where(mask)  # Get valid edges
            del mask 
            gc.collect()
            edges.append(torch.stack([src + i, dst]))  # Offset indices
            del src , dst 
            gc.collect()

        del volume_id , layer_id , hits_spatial 
        gc.collect()
        edge_index = torch.cat(edges, dim=1)
        del edges 
        gc.collect()
        # remove self loops from the edge_index thus generated : 
        edge_index, _ = remove_self_loops(edge_index)
        row , col = edge_index 
        
        # number of edges : 
        num_edges = edge_index.shape[1]
        
        # create edge labels and edge attributes : 
        # Lables : 
            # label == 0 if the two nodes are not part of a traj 
            # label == 1 otherwise 
        edge_labels = ((truth.particle_id.to_numpy()[row] == truth.particle_id.to_numpy()[col]) & ( truth.particle_id.to_numpy()[row] != 0 ))
        edge_labels = Tensor( edge_labels ).float()
        
        # Attributes : 
            # Angle: between the momentum vector of the particle and the displacement vector between the hits. 
            # Distance: euclidean distance between the two hits. 
        pVector = Tensor( truth[['tpx' , 'tpy' ,'tpz']].to_numpy()[row] )
        pVector = pVector/torch.linalg.norm( pVector , ord = 2 , dim = 1 , keepdim= True )
        disp = Tensor( hits[['x','y','z']].to_numpy()[row] -  hits[['x','y','z']].to_numpy()[col] )
        del row , col 
        gc.collect()
        dist = torch.linalg.norm( disp , ord = 2 , dim = 1 , keepdim=True )
        angle = torch.sum( pVector*(disp/dist) , dim = 1 , keepdim=True )
        angle[torch.isnan(angle)] = 0.
        del pVector , disp  
        gc.collect()
        edge_attr = torch.cat([angle , dist] , dim = 1 )
        del angle , dist , hits , truth 
        gc.collect()
        
        # define graph data : 
        graph_data = Data(
            x = node_fets , 
            edge_index=edge_index , 
            edge_attr = edge_attr , 
            label = edge_labels.unsqueeze(1) , 
            num_nodes = nhits , 
            num_edges = num_edges ,
            hit_ids = hit_ids 
        )
        
        return graph_data 
    
    def __len__(self)->int: 
        return self.num_events 
    
    def __getitem__(self,index:int)->Data:
        return self.GraphData(index)
    

def CreateDataLoder(dataset:EventData,batch_size:int,schuffle:bool=True): 
    if batch_size > len(dataset) : 
        raise ValueError('Batch Size cannot be grater than the lenght of the dataset.')
    return DataLoader(dataset=dataset, batch_size=batch_size, shuffle = schuffle)


def train_test_split(
    dataset_path:str, 
    batch_size:int, train_size:int , 
    valid_size:int, 
    num_workers:int=0 
): 
    '''
    num_workers : number of subprocesses to use for data loading. 
    batch_size : number of sampler per batch. 
    train_size , valid_size : relative size of train and validation samples to use. 
    dataset_path : path to the dataset 
    
    Retuens : train, validation and test pytorch_geometric data loders. 
    '''
    # obtain training indices that will be used for validation
    dataset = EventData(path=dataset_path,device=device)
    num_train = len(dataset)
    indices = list(range(num_train))
    
    # shuffle indices
    np.random.shuffle(indices)
    
    train_split = int(np.floor(train_size * num_train))
    valid_split = int(np.floor(valid_size * num_train))

    # get indeces : 
    train_index, valid_index, test_index = indices[0:train_split], indices[train_split:train_split + valid_split], indices[train_split + valid_split:]
    
    # define samplers for obtaining training and validation batches
    train_sampler = SubsetRandomSampler(train_index)
    valid_sampler = SubsetRandomSampler(valid_index)
    test_sampler = SubsetRandomSampler(test_index)
    
    # create data loaders
    train_loader = DataLoader(dataset=dataset, batch_size = batch_size, 
                                            num_workers = num_workers, sampler = train_sampler  )
    valid_loader = DataLoader(dataset=dataset, batch_size = batch_size,
                                            num_workers = num_workers,  sampler = valid_sampler  )
    test_loader = DataLoader(dataset=dataset, batch_size = batch_size,
                                            num_workers = num_workers,  sampler = test_sampler )
    
    return train_loader , valid_loader , test_loader 