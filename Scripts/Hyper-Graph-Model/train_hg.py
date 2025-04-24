if __name__ == "__main__":
    import os 
    import yaml 
    # pytorch 
    import torch 
    # TrackML_HG 
    from TrackML_HG.base import HGCA
    from TrackML_HG.dataset import PCDataset
    # pytorch_lightning
    from pytorch_lightning import Trainer
    from pytorch_lightning.callbacks import ModelCheckpoint,RichProgressBar
    from pytorch_lightning.loggers import TensorBoardLogger
    
    with open( 'Hypergraph-Model.yml' , 'r' ) as f : 
        hparams = yaml.safe_load(f)

    print("PID:", os.getpid())

    torch.cuda.empty_cache()

    torch.set_float32_matmul_precision('high')
    
    script_dir = os.path.dirname(os.path.abspath(__file__))  # Get script directory
    file_path = os.path.join(script_dir, "hparams_embedding.yml")

    with open( file_path , 'r' ) as f : 
        hparams = yaml.safe_load(f)
        
    checkpoint_callback = ModelCheckpoint(
        dirpath=hparams['save_model_path'],
        filename='HG-Model',
        monitor="Loss",  # Metric to monitor (change as needed)
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
    logger = TensorBoardLogger("lightning_logs", name="HG_Model")

    model = HGCA(hparams)
    ds = PCDataset(hparams)

    trainer = Trainer(
        logger  = logger , 
        accelerator = "auto", 
        devices = 2,
        # fast_dev_run = 2, 
        max_epochs=50,
        # limit_train_batches=2,    # Use a small number of batches
        # limit_val_batches=2,
        enable_checkpointing=True, 
        callbacks=[checkpoint_callback,RichProgressBar(),checkpoint_callback2], 
        log_every_n_steps=10,
        enable_progress_bar=True 
    )

    trainer.fit(model , ds )