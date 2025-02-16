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

import torch_geometric
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.utils import  remove_self_loops , scatter 

# make a function that builds a MultiLayerPercerptron : 

def buildMLP( insize:int, outsize:int, features:list, add_bnorm:bool = False, add_activation=None): 
    layers = [] 
    layers.append(nn.Linear( insize , features[0]))
    layers.append( nn.ReLU() )
    for i in range( 1 , len( features ) ): 
        if add_bnorm: 
            layers.append( nn.BatchNorm1d( features[i-1]) )
        layers.append( nn.Linear( features[i-1] , features[i] ) )
        layers.append( nn.ReLU() )
    layers.append(nn.Linear(features[-1],outsize))
    if add_activation != None: 
        layers.append( add_activation )
    return nn.Sequential(*layers)


# Create the meta layer class 
class MetaLayer( torch.nn.Module ): 
    def __init__( self,
                 edge_model: Optional[torch.nn.Module] = None ,
                 node_model: Optional[torch.nn.Module] = None ,
                 global_model: Optional[torch.nn.Module] = None ):
        
        super(MetaLayer, self).__init__()
        self.edge_model = edge_model
        self.node_model = node_model
        self.global_model = global_model
    
        # self.reset_parameters()
    
    def reset_parameters(self):
        """Resets all learnable parameters of the module."""
        for item in [self.node_model, self.edge_model, self.global_model]:
            if hasattr(item, 'reset_parameters'):
                item.reset_parameters()
                
    def forward(
        self,
        x:Tensor, edge_index:Tensor, 
        edge_attr: Optional[Tensor] = None, 
        u : Optional[Tensor] = None, 
        batch : Optional[Tensor] = None
    ) -> Tuple[ Tensor , Optional[Tensor] , Optional[Tensor] ] : 
        
        row , col = edge_index[0] , edge_index[1] 
        
        y =  batch if batch is None else batch[row] 
        # print( x.shape )
        # Edge level step 
        if self.edge_model is not None: 
            edge_attr = self.edge_model( x[row] , x[col] , edge_attr, u ,   y  ) 
        # Node level Step 
        if self.node_model is not None: 
            x = self.node_model(x,edge_index,edge_attr,u,batch) 
        # Graph Level Step 
        if self.global_model is not None: 
            u = self.global_model(x,edge_index,edge_attr,u,batch)  
        
        return x , edge_attr , u 
    
    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}(\n'
                f'  edge_model={self.edge_model},\n'
                f'  node_model={self.node_model},\n'
                f'  global_model={self.global_model}\n'
                f')')
        
# Build the edge network : 

class EdgeNet(nn.Module): 
    def __init__(
        self, in_edge:int, 
        out_edge:int, node_dim:int,
        features:list  , global_dim:Optional[int] = None , 
        add_bnorm:bool = False 
    ):
        super( EdgeNet , self ).__init__()
        if global_dim is None : 
            global_dim = 0 
        self.edge_mlp =  buildMLP(
            insize= 2*node_dim + global_dim +  in_edge , 
            outsize= out_edge , features= features , 
            add_bnorm= add_bnorm
        )
    
    def forward(
        self,src:Tensor,dst:Tensor, 
        edge_attr:Tensor, 
        u:Optional[Tensor]=None, 
        edge_batch:Optional[Tensor]=None
    ): 
        if (u is not None) and (edge_batch is None) : 
            raise ValueError('Must Pass edge_batch if global data is present' )
        
        out = torch.cat([src,dst,edge_attr],dim=1)
        
        if u is not None : 
            out = torch.cat([out,u[edge_batch]],dim=1)
            
        return self.edge_mlp(out)
    
    
# Node update block : 
class NodeNet( torch.nn.Module ): 
    def __init__(
        self, innode:int, outnode:int, 
        inedge:Optional[int] = 0 , 
        inglobal:Optional[int] = 0 ,  
        features:Optional[list] = 0 , 
        add_bnorm:bool = False 
    ): 
        
        super( NodeNet , self ).__init__()
        self.node_mlp = buildMLP(
            insize=innode+inedge+inglobal, 
            outsize=outnode , 
            features=features , 
            add_bnorm= add_bnorm
        )
        
    def forward(
        self , x:Tensor , 
        edge_index:Tensor, 
        edge_attr:Optional[Tensor]=None, 
        u:Optional[Tensor]=None, 
        batch:Optional[Tensor]=None 
    ): 
        if u is not None and batch is None : 
            raise ValueError('Must Pass edge_batch if global data is present' )
        
        _ , col = edge_index 
        
        out = x 
        
        if edge_attr is not None : 
            y = scatter(
                edge_attr , col , dim = 0 , 
                dim_size=x.size(0) , reduce='mean'
            )
            out = torch.cat( [ out , y  ] , dim = 1 )
            del y 
        
        if u is not None : 
            out = torch.cat( out , u[batch] , dim = 1 )
             
        return self.node_mlp( out )


# Global Update Block : 

class GlobalNet( torch.nn.Module ): 
    def __init__(
        self, 
        inglobal:int, outglobal:int, 
        features:list, innode:int , 
        inedge:Optional[int]=0  ,
        add_bnorm:bool = False 
    ): 
        
        super( GlobalNet , self ).__init__() 
        self.global_mlp = buildMLP(
            insize=inedge+innode+inglobal, 
            outsize=outglobal, 
            features=features, 
            add_bnorm=add_bnorm
        )
        
    def forward(
        self, x:Tensor, 
        edge_index:Tensor,
        u:Tensor, batch:Tensor , 
        edge_attr:Optional[Tensor]=None
    ): 
        
        src_idx , _ = edge_index 
        
        out = torch.cat([u, scatter(x,batch,dim=0,reduce='mean')] , dim = 1 )
        if edge_attr is not None : 
            out = torch.cat([u,scatter(edge_attr,batch[src_idx],dim=0,reduce='mean')],dim=1)
        
        return self.global_mlp( out )
    
    
# now we define the full GNN Model using metalayers: 

class GNN_MetaLayer_Model(torch.nn.Module): 
    
    def __init__(
        self , nmeta_layers:int, 
        node_feats:list, 
        inter_node_feats:list, 
        edge_feats:Optional[list]=None, 
        inter_edge_feats:Optional[list]=None, 
        global_feats:Optional[list]=None, 
        inter_global_feats:Optional[list]=None , 
        add_bnorm:bool = False 
    ):
        super(GNN_MetaLayer_Model , self ).__init__()
        
        self.meta_layers = nn.ModuleList([])
        
        if inter_node_feats == None : 
            raise ValueError('Inter Node feats must also be supplied along with node_feats')
        if len( node_feats ) != nmeta_layers + 1 : 
            raise ValueError('The length of \'node_feats\' must be equal to nmeta_layers + 1 ' )
        if len( inter_node_feats ) != nmeta_layers : 
            raise ValueError('The length of \'inter_node_feats\' must be equal to nmeta_layers ' )
        
        if edge_feats != None : 
            if inter_edge_feats == None : 
                raise ValueError('Inter edge feats must also be supplied along with edge_feats')
            if len( edge_feats ) != nmeta_layers + 1 : 
                raise ValueError('The length of \'edge_feats\' must be equal to nmeta_layers + 1 ' )
            if len( inter_edge_feats ) != nmeta_layers  : 
                raise ValueError('The length of \'inter_edge_feats\' must be equal to nmeta_layers ' )
            
        if global_feats != None : 
            if inter_global_feats == None : 
                raise ValueError('Inter global feats must also be supplied along with global_feats')
            if len( global_feats ) != nmeta_layers + 1 : 
                raise ValueError('The length of \'global_feats\' must be equal to nmeta_layers + 1 ' )
            if len( inter_global_feats ) != nmeta_layers : 
                raise ValueError('The length of \'inter_global_feats\' must be equal to nmeta_layers ' )
        
        if (node_feats is None ) and (edge_feats is None ) and (global_feats is None ) : 
            raise ValueError('All Type of Netorks are Null')
        
        self.nmeta = nmeta_layers
        
        edge_part , node_part , global_part = None , None , None 
        for i in range( nmeta_layers ): 
            
            if global_feats is not None : 
                current_global_feat = global_feats[i]
            else : 
                current_global_feat = 0 
    
            if edge_feats is not None : 
                edge_part = EdgeNet(
                    in_edge=edge_feats[i],
                    out_edge=edge_feats[i+1] , 
                    node_dim=node_feats[i] , 
                    global_dim=current_global_feat, 
                    features=inter_node_feats[i] , 
                    add_bnorm=add_bnorm
                )
                current_edge_feat = edge_feats[i+1]
            else : 
                current_edge_feat = 0
                
            if node_feats is not None : 
                node_part = NodeNet(
                    innode=node_feats[i] , 
                    outnode=node_feats[i+1],
                    features=inter_node_feats[i],
                    inedge=current_edge_feat,
                    inglobal=current_global_feat , 
                    add_bnorm=add_bnorm
                )
            if global_feats is not None : 
                global_part = GlobalNet(
                    inedge=current_edge_feat,
                    innode=node_feats[i+1],
                    inglobal=global_feats[i] , 
                    outglobal=global_feats[i+1] , 
                    features=inter_global_feats[i] , 
                    add_bnorm=add_bnorm
                )
            self.meta_layers.append(
                MetaLayer(
                    edge_model=edge_part,
                    node_model=node_part,
                    global_model=global_part 
                )
            )
            
    def forward( self , data ):
        x , edge_attr , global_data = data.x , None , None 
        
        if 'edge_attr' in data : 
            edge_attr = data.edge_attr  
            
        if 'global_data' in data : 
            global_data = data.global_data 
        
        for i in range(self.nmeta) : 
            x , edge_attr , global_data = self.meta_layers[i](
                x = x , edge_index = data.edge_index , 
                edge_attr = edge_attr , u = global_data , batch = data.batch     
            )
            
        # return final updated  featuers ! 
        return x , edge_attr , global_data 