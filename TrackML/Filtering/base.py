# pytorch 
import torch
import torch.nn as nn
# lightning 
from pytorch_lightning import LightningModule
# torchmetrics 
from torchmetrics.classification import BinaryPrecision, BinaryRecall, BinaryF1Score
# TrackML 
from .utils import FilteringModel
from .metrics import LogSumLoss

class FilteringModelPl(LightningModule): 
    
    def __init__(self,hparams): 
        super().__init__()
        self.save_hyperparameters(hparams)
        
        # Metrics (with DDP support)
        self.train_precision = BinaryPrecision()
        self.train_recall = BinaryRecall()
        self.train_f1 = BinaryF1Score()

        self.val_precision = BinaryPrecision()
        self.val_recall = BinaryRecall()
        self.val_f1 = BinaryF1Score()
        
        self.log_sum_loss = LogSumLoss()
        
        self.model = FilteringModel(
            in_features=hparams['in_featuers'],
            hidden_features=hparams['hidden_features']
        )
        
    def forward(self , x ): 
        return self.model( x )
    
    def training_step(self, batch , batch_idx ):

        output = self( batch )
        
        loss_fn = nn.BCEWithLogitsLoss(
            reduction='sum' , pos_weight=batch.edge_purity/( 1-batch.edge_purity)
        )
        loss = loss_fn(output,batch.edge_attr)
        
        with torch.no_grad() : 
            preds = torch.nn.Sigmoid()(output)
            self.train_precision.update( preds , batch.edge_attr )
            self.train_recall.update(preds, batch.edge_attr)
            self.train_f1.update(preds, batch.edge_attr)
            
        if torch.cuda.is_available(): 
                self.log(
                    'Memory Allocated' , torch.cuda.memory_allocated()/(1024**3), 
                    prog_bar=True , on_step = True , on_epoch=True, 
                    reduce_fx='max' , sync_dist=True
                )
        
        self.log('loss' , loss , on_step = True , on_epoch = False , prog_bar=True )
        self.last_idx = batch_idx
        return loss
    
    def on_train_epoch_end(self):
        self.log("train_precision", self.train_precision.compute(), prog_bar=True,sync_dist=True)
        self.log("train_recall", self.train_recall.compute(), prog_bar=True ,sync_dist=True)
        self.log("train_f1", self.train_f1.compute(), prog_bar=True,sync_dist=True)
        
        self.train_precision.reset()
        self.train_recall.reset()
        self.train_f1.reset()
        
        if self.last_idx % 10 == 0 and torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect() 
    
    def  _test_val_common_step_( self , batch , batch_idx ): 
        with torch.no_grad() : 
            output = self( batch )
        
            loss_fn = nn.BCEWithLogitsLoss(
                reduction='sum' , pos_weight=batch.edge_purity/( 1-batch.edge_purity)
            )
            loss = loss_fn(output,batch.edge_attr)
            self.log_sum_loss.update( loss )
            self.log('loss' , loss , on_step = True , on_epoch = False , prog_bar=True )     
            
            if torch.cuda.is_available(): 
                self.log(
                    'Memory Allocated' , torch.cuda.memory_allocated()/(1024**3), 
                    prog_bar=True , on_step = True , on_epoch=True, 
                    reduce_fx='max' , sync_dist=True
                )       
            
            preds = torch.nn.Sigmoid()(output)
            
            self.val_precision.update(preds, batch.edge_attr)
            self.val_recall.update(preds, batch.edge_attr)
            self.val_f1.update(preds, batch.edge_attr)
            
            self.last_idx = batch_idx
            
            return loss 

    def validation_step(self,batch,batch_idx):
        return self._test_val_common_step_(batch,batch_idx)

    def test_step(self,batch,batch_idx): 
        return self._test_val_common_step_(batch,batch_idx)
        
    def on_validation_epoch_end(self):
        self.log("val_precision", self.val_precision.compute(), prog_bar=True ,sync_dist=True)
        self.log("val_recall", self.val_recall.compute(), prog_bar=True ,sync_dist=True)
        self.log("val_f1", self.val_f1.compute(), prog_bar=True ,sync_dist=True)

        self.val_precision.reset()
        self.val_recall.reset()
        self.val_f1.reset()
        self.log_sum_loss.reset()
        
        if self.last_idx % 10 == 0 and torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        
    def configure_optimizers(self): 
        return torch.optim.SGD(self.model.parameters(),lr = self.hparams['lr'])