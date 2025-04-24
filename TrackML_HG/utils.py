from typing import Optional, Tuple
# pytorch 
import torch
import torch.nn as nn
from torch.nn import Linear, Module
from torch import Tensor
# pytorch geometric
from torch_geometric.utils import  softmax 
from torch_geometric.nn import  global_mean_pool 
# torch scatter 
from torch_scatter import scatter_add
# TrackML
from TrackML.Models.utils import buildMLP 


def generate_hyperedges(node_features: Tensor, radius_threshold: float, batch_size: int = 32) -> Tuple[Tensor, int]:
    """
    Generates hyperedge data for input to a hyperconv layer using the radius cluster algorithm.

    Args:
        node_features (Tensor): Node features as a PyTorch tensor of shape [num_nodes, num_features].
        radius_threshold (float): Radius threshold for clustering nodes into hyperedges.
        batch_size (int): Batch size for processing distances.

    Returns:
        Tuple[Tensor, int]: A tuple containing the edge_index tensor and the number of hyperedges.
    """
    num_nodes = node_features.size(0)
    hyperedges = []

    # Generate hyperedges using batching
    for i in range(0, num_nodes, batch_size):
        batch_end = min(i + batch_size, num_nodes)
        batch_features = node_features[i:batch_end]  # Shape: [batch_size, num_features]

        # Compute distances for the batch
        distances = torch.cdist(batch_features, node_features)  # Shape: [batch_size, num_nodes]

        # Identify nodes within the radius threshold
        for j, distance_row in enumerate(distances):
            hyperedge = torch.where(distance_row <= radius_threshold)[0]
            hyperedges.append(hyperedge)

    # Convert hyperedges to edge_index format
    edge_index = []
    for idx, hyperedge in enumerate(hyperedges):
        for node in hyperedge:
            edge_index.append([node.item(), idx])

    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()

    return edge_index, len(hyperedges)


# coustom hypergraph message passing layer with attention score outputs : 
class HypergraphConvWithAttention(nn.Module):
    def __init__(self, in_node_dim, in_edge_dim, out_dim, use_bias=True):
        super().__init__()
        self.node_proj = Linear(in_node_dim, out_dim, bias=use_bias)
        self.edge_proj = Linear(in_edge_dim, out_dim, bias=use_bias)
        self.att_proj = Linear(out_dim, 1, bias=use_bias)
        self.out_proj = Linear(out_dim, out_dim, bias=use_bias)

    def forward(self, x, e, hyperedge_index):
        '''
        x : Node features [N, in_node_dim]
        e : Hyperedge features [M, in_edge_dim]
        hyperedge_index : Edge index [2, E] where E is the number of edges
        output : Node features [N, out_dim] and attention weights [E]
        '''
        row, col = hyperedge_index  # node idx (i), hyperedge idx (j)

        #Project node and hyperedge features
        x_i = self.node_proj(x[row])       # [E, H]
        e_j = self.edge_proj(e[col])       # [E, H]

        #Compute attention weights (tanh + projection)
        att_input = torch.tanh(x_i + e_j)  # [E, H]
        att_scores = self.att_proj(att_input).squeeze(-1)  # [E]

        #Normalize attention weights per node i.e 
        #   across hyperedges per node
        att_weights = softmax(att_scores, index=row)       # [E]
        
        #Weighted message passing from hyperedges to nodes
        messages = e_j * att_weights.unsqueeze(-1)         # [E, H]
        node_updates = scatter_add(messages, row, dim=0, dim_size=x.size(0))  # [N, H]

        # Optional: post-transform (e.g. residual or output layer)
        out = self.out_proj(node_updates)  # [N, H]
        return out, att_weights
    

class HypergraphConvModelWithAttentionOut(nn.Module):
    """
    Builds a PyG Sequential model with HypergraphConvWithAttention layers,
    returning the final node output and attention weights from the last HGConv.

    Args:
        in_node_dim (int): Input node feature dimension.
        in_edge_dim (int): Input hyperedge feature dimension.
        hidden_dims (list[int]): List of hidden dimensions for each HGConv layer.
        out_dim (int): Output feature dimension for final node prediction.

    Returns:
        model: nn.Module that returns (node_output, last_attention_weights)
    """
    def __init__(
        self , 
        in_node_dim: int, in_edge_dim: int, 
        hidden_dims: list[int], out_dim: int
    ):
        super().__init__()
        self.layers = nn.ModuleList()
        self.activations = nn.ModuleList()

        dims = [in_node_dim] + hidden_dims
        for i in range(len(hidden_dims)):
            self.layers.append(HypergraphConvWithAttention(dims[i], in_edge_dim, dims[i+1]))
            self.activations.append(nn.ReLU())

        self.final_proj = HypergraphConvWithAttention( dims[-1] , in_edge_dim, out_dim )

    def forward(self, x, e, hyperedge_index):
        for conv, act in zip(self.layers, self.activations):
            x, _ = conv(x, e, hyperedge_index)
            x = act(x)
        
        return self.final_proj(x , e, hyperedge_index)
    
    
class HGNN(Module):
    def __init__(
        self,
        node_insize: int,
        node_outsize: int,
        node_features: list,
        hpr_edge_outsize: int,
        hpr_edge_features: list ,
        hg_outsize: int,
        hg_features: list
    ):
        super(HGNN, self).__init__()
        # generate mlp layer for node embedding for hypergraph structure 
        self.mlp_node = buildMLP(node_insize, node_outsize, node_features)
        
        # generate mlp to get embedd hyperedge features
        self.mlp_hyperedge = buildMLP(node_insize, hpr_edge_outsize, hpr_edge_features)
        
        # generate hyergraphconvolution network for the hypergraph structure
        self.hgconv = HypergraphConvModelWithAttentionOut(
            in_node_dim=node_outsize + node_insize,
            in_edge_dim=hpr_edge_outsize + node_insize, 
            hidden_dims=hg_features,
            out_dim=hg_outsize
        )
        
    def forward(self, x: Tensor) -> Tensor: 
        # embedd node features
        node_embedding = self.mlp_node(x)
        # generate hyperedge structure form these node embeddings 
        edge_index , _  = generate_hyperedges( node_embedding , radius_threshold=0.1)
        row , col = edge_index
        
        # generate hyperedge features
        # 1. get the hyperedge features from the node features via mean pooling
        hyperedge_features = global_mean_pool(x[row], edge_index[1])
        # embeddd hyperedge features
        hyperedge_features_embdd = self.mlp_hyperedge(hyperedge_features)
        
        # number of nodes and number of hyperedges
        N = x.shape[0]
        M = hyperedge_features.shape[0]
        
        # append original featuers to the hyperedge features
        hyperedge_features = torch.cat([hyperedge_features, hyperedge_features_embdd], dim=-1)
        # append original features to the node features
        x = torch.cat([x, node_embedding], dim=-1)
        
        # run the hypergraph convolution layer
        _ , att_weights = self.hgconv(
            x = x , e = hyperedge_features, hyperedge_index = edge_index
        )
        
        # get the node to edge scores
        node_to_edge_scores = torch.zeros(N , M)
        node_to_edge_scores[row, col] = att_weights
    
        return node_to_edge_scores  
    
    
class EventLossFunction(nn.Module):
    def __init__(self, batch_size: int = 1024):
        """
        Initialize the EventLossFunction with batching support.

        Args:
            batch_size (int): The size of the batches to process pairwise computations.
        """
        super(EventLossFunction, self).__init__()
        self.bce_loss = nn.BCELoss(reduction='sum')  # Use 'sum' to accumulate loss
        self.batch_size = batch_size

    def forward(self, softmax_output: Tensor, labels: Tensor) -> Tensor:
        """
        Compute the binary cross-entropy loss for the given softmax output and labels using batching.

        Args:
            softmax_output (Tensor): Softmax output of shape (num_nodes, num_hyperedges).
            labels (Tensor): Labels of shape (num_nodes).

        Returns:
            Tensor: Computed loss as a single scalar tensor.
        """
        num_nodes, _ = softmax_output.shape
        device = softmax_output.device

        # Initialize loss accumulator
        total_loss = torch.tensor(0.0, device=device)

        # Compute pairwise probabilities and labels in batches
        for i in range(0, num_nodes, self.batch_size):
            end_i = min(i + self.batch_size, num_nodes)
            softmax_batch = softmax_output[i:end_i]  # Shape: (batch_size, num_hyperedges)

            # Pairwise probabilities for the current batch
            pairwise_probs = torch.matmul(softmax_batch, softmax_output.T)  # Shape: (batch_size, num_nodes)

            # Pairwise label agreement for the current batch
            pairwise_labels = (labels[i:end_i].unsqueeze(1) == labels.unsqueeze(0)).float()  # Shape: (batch_size, num_nodes)

            # Mask diagonal (self-loops) to avoid self-comparison
            mask = torch.eye(end_i - i, num_nodes, device=device)
            pairwise_probs = pairwise_probs * (1 - mask)
            pairwise_labels = pairwise_labels * (1 - mask)

            # Flatten and compute BCE loss for the current batch
            batch_loss = self.bce_loss(pairwise_probs.flatten(), pairwise_labels.flatten())
            total_loss += batch_loss

        # Normalize the total loss by the number of pairs
        total_pairs = torch.tensor( num_nodes * (num_nodes - 1) )  # Total number of valid pairs
        return total_loss / total_pairs


def create_disjoint_hypergraph(softmax_output: Tensor) -> dict:
    """
    Create a disjoint hypergraph by assigning each node to its most probable hyperedge.

    Args:
        softmax_output (Tensor): Softmax output of shape (num_nodes, num_hyperedges).

    Returns:
        dict: A dictionary where keys are hyperedge indices and values are tensors of node indices.
    """
    # Find the most probable hyperedge for each node
    most_probable_hyperedges = torch.argmax(softmax_output, dim=-1)
    # Create a disjoint hypergraph by grouping nodes based on their assigned hyperedges
    disjoint_hypergraph = {hyperedge.item(): [] for hyperedge in most_probable_hyperedges.unique()}
    for node, hyperedge in enumerate(most_probable_hyperedges):
        disjoint_hypergraph[hyperedge.item()].append(node)

    # Convert the disjoint hypergraph to a more readable format
    disjoint_hypergraph = {key: torch.tensor(value) for key, value in disjoint_hypergraph.items()}

    return disjoint_hypergraph

