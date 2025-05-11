import os

from torch.utils.data import DataLoader
import torch

from model import NARMWrapper
from torch_dataset import RecsysDataset, RecsysCollateFunction
from configs import load_config

from pytorch_lightning.loggers import MLFlowLogger
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
    ModelSummary,
    StochasticWeightAveraging
)

if __name__ == "__main__":
    # Initialize MLFlow Logger
    seed_everything(42) # Reproducibility purpose
    experiment_name = "init_track_embedding"
    run_name = f"{experiment_name}_e32_l4"
    tracking_uri = f"file:experiments/{experiment_name}"

    mlflow_logger = MLFlowLogger(
        experiment_name=experiment_name,
        run_name=run_name,
        tracking_uri=tracking_uri
    )

    # Config 
    configs = load_config("configs/narm.yml")
    data_cfg = configs["data"]
    train_cfg = configs["training"]

    # Hyperparameters
    hparams = configs["hparams"]
    hparams['lr'] = float(hparams['lr'])
    mlflow_logger.log_hyperparams(hparams)

    # Data source 
    root = data_cfg['root']
    session_data = data_cfg['folder']

    # Initialize dataloader
    train_ds = RecsysDataset(data_path=os.path.join(root, session_data, "train.csv"))
    val_ds   = RecsysDataset(data_path=os.path.join(root, session_data, "val.csv"))
    test_ds  = RecsysDataset(data_path=os.path.join(root, session_data, "test.csv"))

    collate_fn = RecsysCollateFunction(padding_idx=hparams["padding_idx"])

    batch_size = train_cfg["batch_size"]
    train_ld = DataLoader(dataset=train_ds, batch_size=batch_size, shuffle=True, num_workers=4, persistent_workers=True, collate_fn=collate_fn)
    val_ld   = DataLoader(dataset=val_ds, batch_size=batch_size, shuffle=False, num_workers=4, persistent_workers=True, collate_fn=collate_fn)
    test_ld  = DataLoader(dataset=test_ds, batch_size=batch_size, shuffle=False, num_workers=4, collate_fn=collate_fn)

    # Initialize model 
    narm = NARMWrapper(embedding_dim=hparams["embedding_dim"], 
                    hidden_size=hparams["hidden_size"], 
                    n_gru_layers=hparams["n_gru_layers"], 
                    n_items=hparams["n_items"], 
                    padding_idx=hparams["padding_idx"],
                    embedding_matrix_path=hparams["embedding_matrix_path"], 
                    lr=hparams["lr"],
                    k = train_cfg["k"])

    # Callbacks 
    check_point_name = f"best-checkpoint_{experiment_name}"

    training_callbacks = [
            EarlyStopping(monitor="val_loss", mode="min", patience=train_cfg['patience']),
            StochasticWeightAveraging(swa_lrs=1e-2),
            LearningRateMonitor(logging_interval="step"),
            ModelCheckpoint(
                monitor="val_loss",
                mode="min",
                save_top_k=1,
                dirpath="checkpoints/",
                filename=f"best-checkpoint_{run_name}"
            ),
            ModelSummary(-1)    
        ]

    # Training
    torch.cuda.empty_cache()  
    trainer = Trainer(
        logger=mlflow_logger,
        callbacks=training_callbacks,
        max_epochs=train_cfg["epochs"],
        log_every_n_steps=1,
    )

    trainer.fit(model=narm,
                train_dataloaders=train_ld,
                val_dataloaders= val_ld, 
                ckpt_path=None)

    # Testing 
    trainer.test(model=narm, 
            dataloaders=test_ld, 
            ckpt_path=None)

