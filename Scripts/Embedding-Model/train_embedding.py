if __name__ == "__main__":
    import os
    import yaml
    import torch  
    # pytorch lightning 
    from pytorch_lightning import Trainer
    from pytorch_lightning.callbacks import ModelCheckpoint,RichProgressBar
    from pytorch_lightning.loggers import TensorBoardLogger
    # TrackML Packages 
    from TrackML.Embedding.base import EmbeddingBase
    from TrackML.Embedding.dataset import EmbeddingDataset

    torch.cuda.empty_cache()

    torch.set_float32_matmul_precision('high')
    
    script_dir = os.path.dirname(os.path.abspath(__file__))  # Get script directory
    file_path = os.path.join(script_dir, "hparams_embedding.yml")

    with open( file_path , 'r' ) as f : 
        hparams = yaml.safe_load(f)
        
    checkpoint_callback = ModelCheckpoint(
        dirpath=hparams['save_model_path'],
        filename='Embedding-Model',
        monitor="Log Loss",  # Metric to monitor (change as needed)
        save_top_k=1,  # Save only the best model
        mode="min",  # "min" for loss, "max" for accuracy or other metrics
        verbose=True,
        every_n_train_steps = 0,
        every_n_epochs = 1
    )
    
    # Initialize TensorBoard logger
    logger = TensorBoardLogger("lightning_logs", name="Embedding_Model")

    model = EmbeddingBase(hparams)
    ds = EmbeddingDataset(hparams)


    trainer = Trainer(
        logger  = logger , 
        accelerator = "auto", 
        devices = 1,
        # fast_dev_run = 2, 
        max_epochs=100,
        # limit_train_batches=2,    # Use a small number of batches
        # limit_val_batches=2,
        enable_checkpointing=True, 
        callbacks=[checkpoint_callback,RichProgressBar()], 
        log_every_n_steps=20,
        enable_progress_bar=True 
    )

    trainer.fit(model , ds )