# pytorch 
import torch
# pytorch_lightning
from pytorch_lightning import LightningModule
# TrackML_HG
from TrackML_HG.metrics import ParticlePurity, TrackPurity
from TrackML_HG.utils import * 

class HGCA(LightningModule): 
    
    def __init__(self,hparams): 
        super().__init__()
        self.save_hyperparameters(hparams)
        
        # Metrics (with DDP support)
        self.particle_purity = ParticlePurity()
        self.track_purity = TrackPurity()
        
        # Losses
        self.loss = EventLossFunction()
        
        self.model = HGNN(
            node_insize = self.hparams['node_insize'],
            node_outsize = self.hparams['node_outsize'],
            node_features = self.hparams['node_features'],
            hpr_edge_outsize = self.hparams['hpr_edge_outsize'],
            hpr_edge_features = self.hparams['hpr_edge_features'],
            hg_outsize = self.hparams['hg_outsize'],
            hg_features = self.hparams['hg_features']
        )
        
        self.last_idx = 0
        
    def forward(self , x ): 
        return self.model( x )
    
    def training_step(self , batch , batch_idx ): 
        
        event_data , labels = batch 
        event_data.squeeze_(dim=0)
        labels.squeeze_(dim=0)
        
        output = self( event_data )
        
        loss = self.loss( output , labels )
        self.log(
            'Loss' , loss, 
            prog_bar = True , on_step = True , on_epoch = False 
        )
        
        if torch.cuda.is_available():
            self.log(
                'Memory Allocated' , torch.cuda.memory_allocated()/(1024**3), 
                prog_bar=True , on_step = True , on_epoch=True, 
                reduce_fx='max'
            )
        self.last_idx = batch_idx
        return loss 
    
    def on_train_batch_end(self, outputs, batch, batch_idx):
        if self.last_idx % 10 == 0 and torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect() 
    
    # common logic for test and validation 
    def _test_val_common_step_(self , batch , batch_idx ): 
        
        with torch.no_grad(): 
            event_data , labels = batch 
            event_data.squeeze_(dim=0)
            labels.squeeze_(dim=0)
            
            output = self( event_data )
            
            loss = self.loss( output , labels ) 
            
            disjoint_hypergraph = create_disjoint_hypergraph(output)
            self.particle_purity.update(disjoint_hypergraph, labels)
            self.track_purity.update(disjoint_hypergraph, labels)
            
            self.last_idx = batch_idx
            
            return loss 
    
    def validation_step(self,batch,batch_idx):
        return self._test_val_common_step_(batch,batch_idx)
    
    def test_step(self,batch,batch_idx): 
        return self._test_val_common_step_(batch,batch_idx)
    
    def on_validation_batch_end(self, outputs, batch, batch_idx, dataloader_idx = 0):
        if self.last_idx % 10 == 0 and torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    
    def on_validation_epoch_end(self):
        self.log_dict({
            "Particle Purity": self.particle_purity.compute(),
            "Track Purity": self.track_purity.compute(),
        }, prog_bar=True)
        self.particle_purity.reset()
        self.track_purity.reset()

        
    def on_test_epoch_end(self):
        self.log_dict({
            "Particle Purity": self.particle_purity.compute(),
            "Track Purity": self.track_purity.compute(),
        }, prog_bar=True)
        self.particle_purity.reset()
        self.track_purity.reset()
    
    def on_test_batch_end(self, outputs, batch, batch_idx, dataloader_idx = 0):
        if self.last_idx % 10 == 0 and torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    
    # set the optimizer  :
    def configure_optimizers(self): 
        return torch.optim.SGD(self.model.parameters(),lr = self.hparams['lr'])