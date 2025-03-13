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
    if add_activation is not None: 
        layers.append( add_activation )
    return nn.Sequential(*layers)