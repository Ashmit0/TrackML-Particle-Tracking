import os
# Pytorch 
import torch
# Pytorch Lightning
import pytorch_lightning as pl
# Parent Module Imports : 
from TrackML import Preprocessing
from .utils import PointCloudData,train_test_split


### Define pytorch lightning dataset : 
class EmbeddingDataset(pl.LightningDataModule): 
    
    # initialize the class : 
    def __init__(self,hparams)->None: 
        super().__init__() 
        self.save_hyperparameters(hparams)
        
    # def prepare_data(self)->None: 
        # self.detector = Preprocessing.load_detector_data(self.hparams['detector_path'])
        # get the list of event ids from the dataset folder : 
        # self.eventids = [ code[:-9] for code in os.listdir(self.hparams['dataset_path']) if code.endswith('-hits.csv') ]
        # self.dataset = PointCloudData(dataset_path=self.hparams['dataset_path'] , detector_path=self.hparams['detector_path'] , min_nhits=self.hparams['min_hits'] )
    
    def setup(self,stage=None)->None: 
        self.dataset = PointCloudData(dataset_path=self.hparams['dataset_path'] , detector_path=self.hparams['detector_path'] , min_nhits=self.hparams['min_hits'] , max_r = self.hparams['max_r'] , drop_fake= self.hparams['drop_fake'] )
        self.train_ds , self.val_ds , self.test_ds = train_test_split(
            dataset=self.dataset, valid_size=self.hparams['valid_size'], 
            test_size=self.hparams['test_size'], num_works=self.hparams['num_works']
        )
        
    def train_dataloader(self): 
        return self.train_ds 
    def val_dataloader(self): 
        return self.val_ds 
    def test_dataloader(self): 
        return self.test_ds