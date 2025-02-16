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

from .Dataset import EventData

# we write train and test functions : 
def train_pass(model, train_loader, optimizer)->float:
    
    train_loss_ep , data_pts = 0. , 0 
    
    model.train()
    for _ , data in enumerate(train_loader):
        # data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        _ , edge_attr , _  = model(data)
        loss = F.binary_cross_entropy_with_logits(edge_attr, data.label , reduction='sum' )
        loss.backward()
        optimizer.step()
        print( loss )
        train_loss_ep += loss.item() 
        print( train_loss_ep )
        data_pts += data.num_edges 
        
    return train_loss_ep/data_pts

# function to test the model: 
def test_pass(model, test_loader)->float:
    
    test_loss_ep  , data_pts = 0. , 0 
    
    model.eval()
    for _ , data in enumerate(test_loader):
        # data, target = data.to(device), target.to(device)
         
        _ , edge_attr , _  = model(data)
        loss = F.binary_cross_entropy_with_logits(edge_attr, data.label , reduction='sum' )
        
        test_loss_ep += loss.item() 
        data_pts += data.num_edges 
        
    return test_loss_ep/data_pts 




def train(
    model , 
    n_epochs:int,
    train_loader:torch_geometric.loader.dataloader.DataLoader , 
    valid_loader:torch_geometric.loader.dataloader.DataLoader, 
    optimizer , 
    model_save_path:Optional[str]=None 
):
    '''
        n_epochs : number of epochs to train the model
        model : Model to train on. 
        train_loder , valid_loader : traing and validation data loders. 
        optimizer: optimizer to use for model training.  
        model_save_path: If supplied saves the best model in the supplied path. 
    '''
    
    # initialize tracker for minimum validation loss
    valid_loss_min = np.inf  # set initial "min" to infinity
    
    for epoch in range(n_epochs):
        train_loss = train_pass(model, train_loader, optimizer)
        valid_loss = test_pass(model, valid_loader)
        
        # print training/validation statistics 
        
        print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(
            epoch+1, train_loss, valid_loss
        ))
        
        # save model if validation loss has decreased
        if valid_loss <= valid_loss_min:
            print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format( valid_loss_min, valid_loss))
            if model_save_path is not None : 
                torch.save(model.state_dict(), model_save_path )
            valid_loss_min = valid_loss