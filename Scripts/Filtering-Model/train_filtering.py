if __name__ == "__main__":
    import os
    import yaml
    import torch  
    # pytorch lightning 
    from pytorch_lightning import Trainer
    from pytorch_lightning.callbacks import ModelCheckpoint,RichProgressBar
    from pytorch_lightning.loggers import TensorBoardLogger
    # TrackML Packages 
    from TrackML.Filtering.base import FilteringModelPl
    from TrackML.Filtering.dataset import FilteringDataset
    
    # print("PID:", os.getpid())

    # torch.cuda.empty_cache()

    # torch.set_float32_matmul_precision('high')
    
    script_dir = os.path.dirname(os.path.abspath(__file__))  # Get script directory
    file_path = os.path.join(script_dir, "hparams_filtering.yml")

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
        every_n_epochs = 1, 
        save_weights_only=True 
    )
    
    
    checkpoint_callback2 = ModelCheckpoint(
        dirpath=hparams['save_checkpoint_path'],
        filename="{epoch}-{step}",  # Include epoch and step in the filename
        save_top_k=-1,
        every_n_train_steps=10,
        save_weights_only=False,
    )
    # Initialize TensorBoard logger
    logger = TensorBoardLogger("lightning_logs", name="Embedding_Model")

    model = FilteringModelPl(hparams)
    ds = FilteringDataset(hparams)

    trainer = Trainer(
        logger  = logger , 
        accelerator = "cpu", 
        fast_dev_run=2 , 
        # devices = 2,
        # fast_dev_run = 2, 
        # max_epochs=50,
        # limit_train_batches=2,    # Use a small number of batches
        # limit_val_batches=2,
        enable_checkpointing=False, 
        # callbacks=[checkpoint_callback,RichProgressBar(),checkpoint_callback2], 
        # log_every_n_steps=10,
        enable_progress_bar=True 
    )

    trainer.fit(model , ds )