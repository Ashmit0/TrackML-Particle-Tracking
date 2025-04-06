# lightining 
import pytorch_lightning as pl
# TrackML 
from .utils import train_test_split , PostEmbeddingGraph

class FilteringDataset(pl.LightningDataModule): 
    def __init__(self,hparams:dict)->None: 
        super().__init__()
        self.save_hyperparameters(hparams)
        
    def setup(self,stage=None): 
        dataset = PostEmbeddingGraph(
            dataset_path=self.hparams['dataset_path'], 
            detector_path=self.hparams['detector_path'], 
            embd_model_path=self.hparams['embd_model_path'], 
            min_nhits = self.hparams['min_nhits'], 
            margin = self.hparams['margin'] , 
            max_num_neighbors= self.hparams['max_num_neighbours']
        )
        self.train_ds , self.val_ds , self.test_ds = train_test_split(
            dataset=dataset, 
            valid_size=self.hparams['valid_size'], 
            test_size=self.hparams['test_size'] , 
            num_works=self.hparams['num_works']
        )
    
    def train_dataloader(self): 
        return self.train_ds 
    def val_dataloader(self): 
        return self.val_ds 
    def test_dataloader(self): 
        return self.test_ds