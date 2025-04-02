import yaml 
# pytorch lightning 
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import DeviceStatsMonitor
# TrackML Packages 
from TrackML.Embedding.base import EmbeddingBase
from TrackML.Embedding.dataset import EmbeddingDataset

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