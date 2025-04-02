
if __name__ == "__main__":
    import os
    import yaml 
    # pytorch lightning 
    from pytorch_lightning import Trainer
    from pytorch_lightning.callbacks import DeviceStatsMonitor
    # TrackML Packages 
    from TrackML.Embedding.base import EmbeddingBase
    from TrackML.Embedding.dataset import EmbeddingDataset


    script_dir = os.path.dirname(os.path.abspath(__file__))  # Get script directory
    file_path = os.path.join(script_dir, "hparams_embedding.yml")

    with open( file_path , 'r' ) as f : 
        hparams = yaml.safe_load(f)

    model = EmbeddingBase(hparams)
    ds = EmbeddingDataset(hparams)

    device_stats = DeviceStatsMonitor()

    trainer = Trainer(
        accelerator = "cpu", 
        devices = "auto",
        enable_checkpointing=False, 
        fast_dev_run = 3, 
        callbacks=[device_stats]
    )

    trainer.fit(model , ds )