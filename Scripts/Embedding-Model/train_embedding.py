import os
import time
import random
import numpy as np

from scipy.stats import ortho_group

from typing import Optional, Tuple


import torch
import torch.nn as nn
from torch.linalg import norm 
import torch.utils.data as data
import torch.nn.functional as F
from torch.nn import Linear, ReLU, BatchNorm1d, Module, Sequential
from torch import Tensor

# torch.set_default_dtype(torch.float64)

from torch_geometric.typing import (
    Adj,
    OptPairTensor,
    OptTensor,
    Size,
    SparseTensor,
    torch_sparse,
)

import torch_geometric
from torch_geometric.data import Data
from torch_geometric.data import Batch
import torch_geometric.transforms as T
from torch.utils.data.sampler import SubsetRandomSampler
from torch_geometric.utils import remove_self_loops, to_dense_adj, dense_to_sparse, is_undirected , to_undirected, contains_self_loops , to_networkx
from torch_geometric.loader import DataLoader
from torch_geometric.nn import MessagePassing, global_mean_pool, knn_graph
from torch_geometric.datasets import QM9
# from torch_scatter import scatter
# from torch_cluster import knn

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
# import uproot
import vector
vector.register_awkward()
import awkward as ak


import pytorch_lightning as pl
from pytorch_lightning import LightningModule
from torchmetrics import Metric

import yaml 

from TrackML.Embedding.utils import PointCloudData
from pytorch_lightning import Trainer
from TrackML.Embedding.base import EmbeddingBase
from TrackML.Embedding.dataset import EmbeddingDataset
from pytorch_lightning.callbacks import DeviceStatsMonitor

with open('hparams_embedding.yml' , 'r' ) as f : 
    hparams = yaml.safe_load(f)

model = EmbeddingBase(hparams)
ds = EmbeddingDataset(hparams)

device_stats = DeviceStatsMonitor()


trainer = Trainer(
    accelerator = "auto", 
    devices = "auto",
    enable_checkpointing=False, 
    fast_dev_run = 3, 
    callbacks=[device_stats]
)

print(f'Model Trainer Details : \n {trainer} ')

trainer.fit(model , ds )