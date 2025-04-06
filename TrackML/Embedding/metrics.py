# Pytorch 
import torch
# torchmetrics 
from torchmetrics import Metric


class LogSumLoss(Metric):
    def __init__(self):
        super().__init__()
        self.add_state("log_losses", default=torch.tensor([]), dist_reduce_fx="cat")

    def update(self, loss):
        # Expect loss to be a tensor of shape (N,) or scalar
        loss = loss.flatten().detach()
        self.log_losses = torch.cat([self.log_losses, torch.log(loss)])

    def compute(self):
        # log-sum-exp: log(sum(exp(x))) = max + log(sum(exp(x - max)))
        if self.log_losses.numel() == 0:
            return torch.tensor(float("-inf"), device=self.log_losses.device)

        max_log = torch.max(self.log_losses)
        return max_log + torch.log(torch.sum(torch.exp(self.log_losses - max_log)))

  
### define the purity metric class for 
# track purity and particle purity 
class Purity(Metric):
    def __init__(self): 
        super().__init__()
        self.add_state(
            'purity', 
            default = torch.tensor(0.0), 
            dist_reduce_fx = 'sum'
        )
        self.add_state(
            'num_events', 
            default = torch.tensor(0.0),
            dist_reduce_fx = 'sum'
        )
        
    def update(self, intersections, num_hits):
        # Ensure tensors are 1D
        self.purity += torch.mean( intersections/num_hits ).to(self.purity.device)
        self.num_events += 1 

    def compute(self):
        # Standard purity: sum of max class counts / total number of samples
        return self.purity/self.num_events