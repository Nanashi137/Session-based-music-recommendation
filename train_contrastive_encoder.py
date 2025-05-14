import os

from torch.utils.data import DataLoader
import torch

from model import ContrastiveEncoderWrapper
from torch_dataset import ContrastiveDataset
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
    experiment_name = "contrastive_encoder"
    run_name = f"{experiment_name}_e64" #e32 = embedding 32 dimension
    tracking_uri = f"file:experiments/{experiment_name}"

    mlflow_logger = MLFlowLogger(
        experiment_name=experiment_name,
        run_name=run_name,
        tracking_uri=tracking_uri
    )

    # Config 
    configs = load_config("configs/ce.yml")
    data_cfg = configs["data"]
    train_cfg = configs["training"]

    # Hyperparameters
    hparams = configs["hparams"]
    hparams['lr'] = float(hparams['lr'])
    mlflow_logger.log_hyperparams(hparams)

    # Data source 
    root = data_cfg['root']
    folder = data_cfg['folder']

    # Initialize dataloader
    train_ds = ContrastiveDataset(data_path=os.path.join(root, folder, "train.csv"), id2features_matrix_path=hparams["id2features_matrix_path"])
    val_ds   = ContrastiveDataset(data_path=os.path.join(root, folder, "val.csv"), id2features_matrix_path=hparams["id2features_matrix_path"])

    batch_size = train_cfg["batch_size"]
    train_ld = DataLoader(dataset=train_ds, batch_size=batch_size, shuffle=True, num_workers=4, persistent_workers=True)
    val_ld   = DataLoader(dataset=val_ds, batch_size=batch_size, shuffle=False, num_workers=4, persistent_workers=True)

    # Initialize model 
    ce = ContrastiveEncoderWrapper(input_dim=hparams["input_dim"], 
                                embedding_dim=hparams["embedding_dim"],
                                learning_rate=hparams["lr"],
                                margin=hparams["margin"])

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
                dirpath="ce_checkpoints/",
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

    trainer.fit(model=ce,
                train_dataloaders=train_ld,
                val_dataloaders= val_ld, 
                ckpt_path=None)

