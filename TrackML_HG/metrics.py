import torch
from torch import Tensor
from torchmetrics import Metric


class ParticlePurity(Metric):
    def __init__(self):
        super().__init__()
        self.add_state("total_purity", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("num_events", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, disjoint_hypergraph: dict, labels: Tensor):
        """
        Update the metric state with a new disjoint hypergraph and corresponding labels.

        Args:
            disjoint_hypergraph (dict): A dictionary where keys are hyperedge indices and values are tensors of node indices.
            labels (Tensor): A tensor of shape (num_nodes,) containing the labels for each node.
        """
        event_purity = 0.0
        num_particles = len( torch.unique(labels) )

        for _, nodes in disjoint_hypergraph.items():
            node_labels = labels[nodes]
            most_common_label = torch.mode(node_labels).values
            intersection = (node_labels == most_common_label).sum().item()

            num_particles_with_most_common_label = (labels == most_common_label).sum().item()
            
            if intersection >= 0.5 * len(node_labels) and intersection >= 0.5 * num_particles_with_most_common_label:
                event_purity += intersection / num_particles_with_most_common_label

        self.total_purity += event_purity / num_particles
        self.num_events += 1

    def compute(self):
        """
        Compute the average particle purity over all events.

        Returns:
            Tensor: The average particle purity.
        """
        return self.total_purity / self.num_events
    
    
class TrackPurity(Metric):
    def __init__(self):
        super().__init__()
        self.add_state("total_purity", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("num_events", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, disjoint_hypergraph: dict, labels: Tensor):
        """
        Update the metric state with a new disjoint hypergraph and corresponding labels.

        Args:
            disjoint_hypergraph (dict): A dictionary where keys are hyperedge indices and values are tensors of node indices.
            labels (Tensor): A tensor of shape (num_nodes,) containing the labels for each node.
        """
        event_purity = 0.0
        num_tracks = len( disjoint_hypergraph )

        for _, nodes in disjoint_hypergraph.items():
            node_labels = labels[nodes]
            most_common_label = torch.mode(node_labels).values
            intersection = (node_labels == most_common_label).sum().item()

            num_particles_with_most_common_label = (labels == most_common_label).sum().item()
            
            if intersection >= 0.5 * len(node_labels) and intersection >= 0.5 * num_particles_with_most_common_label:
                event_purity += intersection / len(node_labels)

        self.total_purity += event_purity / num_tracks
        self.num_events += 1

    def compute(self):
        """
        Compute the average particle purity over all events.

        Returns:
            Tensor: The average particle purity.
        """
        return self.total_purity / self.num_events
    
    
