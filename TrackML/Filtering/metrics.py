
#pytorch 
import torch
# PyG
from torch_geometric.data import Data
# toechmetrics 
from torchmetrics import Metric
# TrackML
from .utils import get_disconnected_components

def event_reconstruction_metrics( graph_data:Data  )->tuple: 
    '''
        data : graph dataset to get the metrics for : 
        labels : particle_ids for each of the nodes (hits)
        returns : the trajecteory and particle purity of the graph dataset. 
    '''
    
    labels = graph_data.y 
    
    # get the disconnected compoenents (reconstructed tracks) 
    # form the graph steructure : 
    disconnected_components = get_disconnected_components(graph_data)
    
    #  partices with the max occourence for each of the disconnected components : 
    max_track_particle = torch.tensor([ torch.mode(labels[component])[0] for component in disconnected_components])
    # frequency of the particle with the most occourence for each of the disconnected component : 
    max_track_particle_freq = torch.tensor([ torch.sum(labels[track] == particle) for particle,track in zip(max_track_particle,disconnected_components)])
    # get the number of hits that belong to each reconstructed track : 
    num_hits_tracks = torch.tensor( [component.shape[0] for component in disconnected_components] )
    # total number of true hits left by the underlying max_track_particle : 
    max_track_particle_num_true_hits = torch.tensor([torch.sum(labels==particle) for particle in max_track_particle])
    
    # 2. Get the Track Purity : 
    track_purity  = torch.mean(max_track_particle_freq/num_hits_tracks)
    # 3. Get Particle Purity : 
    particle_purity = torch.mean(max_track_particle_freq/max_track_particle_num_true_hits)
    
    return  track_purity , particle_purity 


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