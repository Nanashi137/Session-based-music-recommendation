import pytorch_lightning as L
from pytorch_lightning.utilities.grads import grad_norm

from torch.optim.lr_scheduler import StepLR
import torch 
import torch.nn as nn
import numpy as np

from .narm import NARM 


class NARMWrapper(L.LightningModule): 
    def __init__(self, embedding_dim, hidden_size, n_gru_layers, n_items, padding_idx: int=None, embedding_matrix_path: str=None, lr: float=1e-3):
        super().__init__()
        self.save_hyperparameters()

        self.lr = lr
        self.criterion = nn.CrossEntropyLoss(reduction="mean", ignore_index=n_items if not padding_idx else padding_idx)

        self.embedding_matrix = self._load_embedding_matrix(embedding_matrix_path=embedding_matrix_path) if embedding_matrix_path else None
        self.model = NARM(embedding_dim=embedding_dim, hidden_size=hidden_size, n_gru_layers=n_gru_layers, n_items=n_items, padding_idx=padding_idx, embedding_matrix=self.embedding_matrix)

    def _load_embedding_matrix(self, embedding_matrix_path):
        torch_embedding_matrix = torch.tensor(np.load(file=embedding_matrix_path).astype(np.float32))
        torch_embedding_matrix = torch_embedding_matrix / (torch_embedding_matrix.norm(p=2, dim=1, keepdim=True) + 1e-8)
        return torch_embedding_matrix


    def forward(self, seq, seq_len):
        return self.model(seq, seq_len)
    
    def training_step(self, batch, batch_idx):

        seqs, seq_lens, targets = batch
        logits = self(seqs, seq_lens)

        loss = self.criterion(logits, targets)

        self.log("train_loss", loss, prog_bar=True, on_epoch=True, logger=True)
        
        return loss
        

    def validation_step(self, batch, batch_idx):
        loss = self._shared_eval(batch)
        self.log("val_loss", loss, prog_bar=True, on_epoch=True, logger=True)
        return {"val_loss": loss}
    
    def test_step(self, batch, batch_idx):
        loss, _ = self._shared_eval(batch=batch)
        self.log_dict({"test_loss": loss})
    
    def _shared_eval(self, batch):
        seqs, seq_lens, targets = batch
        logits = self(seqs, seq_lens)

        loss = self.criterion(logits, targets)
        
        return loss

    def on_before_optimizer_step(self, optimizer):
        # Compute the 2-norm for each layer
        # If using mixed precision, the gradients are already unscaled here
        norms = grad_norm(self, norm_type=2)
        self.log_dict(norms)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam([
            {"params": self.model.parameters(), "lr": self.lr}
        ])

        scheduler = StepLR(optimizer=optimizer, step_size=5, gamma=0.8)

        return {
            "optimizer": optimizer,
            "lr_scheduler":{
                "scheduler": scheduler,
                "monitor": "val_loss",
            },
        }
