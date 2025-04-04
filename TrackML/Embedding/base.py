# Pytorch
import torch
# Pytorch Geometric 
import torch_geometric
from torch_geometric.data import Data
# Pytorch Lightning
from pytorch_lightning import LightningModule
# Parent Module Imports 
from .metrics import LogSumLoss,Purity
from .utils import EmbeddingModel,EmbeddingLossFunction,event_reconstruction_metrics

# get the decvice in use 
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


class EmbeddingBase(LightningModule):
    
    # initialize the Embedding Class 
    def __init__(self , hparams ): 
        super().__init__()
        
        # save the hypermeters : 
        self.save_hyperparameters(hparams)
        
        # loss accumulation metric : 
        self.log_sum_loss = LogSumLoss()
        # track and particle purity : 
        self.particle_purity = Purity() 
        self.track_purity = Purity()
        
        # save the model descriptions : 
        self.model = EmbeddingModel(
            in_features=hparams['in_featuers'] , 
            hidden_features=hparams['hidden_featuers'], 
            out_features=hparams['out_featuers']
        )
    
    # forward function 
    def forward(self , inputs ): 
        return self.model( inputs )
    
    # training logic : 
    def training_step(self , batch , batch_idx ): 
        
        event_data , labels = batch 
        event_data.squeeze_(dim=0)
        labels.squeeze_(dim=0)
        
        output = self( event_data )
        
        radius_graph_edge_index = torch_geometric.nn.pool.radius_graph(
            output, 
            r = self.hparams['margin'] , 
            loop = False , 
            max_num_neighbors=self.hparams['max_num_neighbours']
        )
        
        loss = EmbeddingLossFunction(
            x = output,
            labels=labels.squeeze_(dim=0),
            radius_graph_edge_index=radius_graph_edge_index, 
            margin=self.hparams['margin']
        )
        
        self.log('Loss' , loss, prog_bar = True , on_step = True , on_epoch = False )
        self.log_sum_loss.update(loss)
        self.log(
            'Log Loss' , self.log_sum_loss.compute(), on_step = True , 
            on_epoch = True , prog_bar = True , reduce_fx = 'max' 
        )
        
        if device == torch.device('cuda'): 
            self.log(
                'Memory Allocated' , torch.cuda.memory_allocated()/(1024**3), 
                prog_bar=True , on_step = True , on_epoch=True, 
                reduce_fx='max'
            )
        
        return loss 
    
    # common logic for test and validation 
    def _test_val_common_step_(self , batch , batch_idx ): 
        
        with torch.no_grad(): 
            event_data , labels = batch 
            event_data.squeeze_(dim=0)
            labels.squeeze_(dim=0)
            
            output = self( event_data )
            
            radius_graph_edge_index = torch_geometric.nn.pool.radius_graph(
                output, 
                r = self.hparams['margin'] , 
                loop = False , 
                max_num_neighbors=self.hparams['max_num_neighbours']
            )
            
            loss = EmbeddingLossFunction(
                x = output,
                labels=labels.squeeze_(dim=0),
                radius_graph_edge_index=radius_graph_edge_index, 
                margin=self.hparams['margin']
            )
            self.log(
                'Loss',loss, prog_bar = True , 
                on_step = True , on_epoch = False 
            )
            self.log_sum_loss.update(loss)
            self.log(
                'Log Loss' , self.log_sum_loss.compute(), 
                on_step = True , on_epoch = True , 
                prog_bar = True , reduce_fx = 'max' 
            )
            
            # create the graph data structure : 
            event_graph_data = Data( x = event_data, edge_index=radius_graph_edge_index, y = labels )
            intersections , num_track , num_particle = event_reconstruction_metrics(event_graph_data,labels)
            
            self.track_purity.update( intersections = intersections , num_hits = num_track )
            self.particle_purity.update( intersections = intersections , num_hits = num_particle )
            
            self.log_dict(
                {'track purity' : self.track_purity.compute() , 'particle purity' : self.particle_purity.compute() }, 
                on_step = True , on_epoch = True , prog_bar = True , reduce_fx = 'mean'
            )
            if device == torch.device('cuda'): 
                self.log(
                    'Memory Allocated' , torch.cuda.memory_allocated()/(1024**3), 
                    prog_bar=True , on_step = True , on_epoch=True, 
                    reduce_fx='max'
                )
            
        return loss
    
    def validation_step(self,batch,batch_idx):
        return self._test_val_common_step_(batch,batch_idx)
    
    def on_validation_epoch_end(self):
        self.track_purity.reset()
        self.particle_purity.reset()
    
    def test_step(self,batch,batch_idx): 
        return self._test_val_common_step_(batch,batch_idx)
    
    # set the optimizer  :
    def configure_optimizers(self): 
        return torch.optim.SGD(self.model.parameters(),lr = self.hparams['lr'])