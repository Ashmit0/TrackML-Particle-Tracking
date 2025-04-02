# Pytorch 
import torch
# torchmetrics 
from torchmetrics import Metric

### define custom log loss metric for pytorch lightning : 
def _custom_dist_reduce_fn(x): 
    return torch.log( torch.sum( x , dim = 0 ) )

class LogSumLoss(Metric):
    def __init__(self):
        super().__init__()
        self.add_state("log_loss_sum", default=torch.tensor(0.0), dist_reduce_fx=_custom_dist_reduce_fn)

    def update(self, loss):
        self.log_loss_sum = torch.log(loss) + torch.log( 1 + torch.exp(self.log_loss_sum)/loss)

    def compute(self):
        return self.log_loss_sum

  
### define the purity metric class for 
# track purity and particle purity 
class Purity(Metric):
    def __init__(self): 
        super().__init__()
        self.add_state(
            'intersections', 
            default = torch.tensor([]), 
            dist_reduce_fx = 'cat'
        )
        self.add_state(
            'num_hits', 
            default = torch.tensor([]),
            dist_reduce_fx = 'cat'
        )
        
    def update(self,intersections,num_hits):
        self.intersections = intersections
        self.num_hits = num_hits
    
    def compute(self): 
        return torch.mean( self.intersections/self.num_hits )