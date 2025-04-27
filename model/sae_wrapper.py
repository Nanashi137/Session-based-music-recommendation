import pytorch_lightning as L
from pytorch_lightning.utilities.grads import grad_norm

from torch.optim.lr_scheduler import StepLR
import torch 
import torch.nn as nn

from .sae import SAE

class SAE_wrapper(L.LightningModule): 
    def __init__(self, dim_list: list, lr: float=3e-3):
        super().__init__()

        self.model = SAE(dim_list=dim_list)
        self.lr = lr

        self.criterion = nn.MSELoss()

    def forward(self, x):
        return self.model(x)
    
    def encode(self, x): 
        return self.model.encode(x)
    
    def decode(self, x): 
        return self.model.decode(x)

    def training_step(self, batch, batch_idx):
        reconstruct = self.forward(batch)
        loss = self.criterion(reconstruct, batch)

        self.log("train_loss", loss, prog_bar=True, on_epoch=True, logger=True)
        
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self._shared_eval(batch=batch)
        self.log("val_loss", loss, prog_bar=True, on_epoch=True, logger=True)
        return {"val_loss": loss}
    
    def test_step(self, batch, batch_idx):
        loss = self._shared_eval(batch=batch)
        self.log_dict({"test_loss": loss})
    
    def _shared_eval(self, batch):
        reconstruct = self.forward(batch)
        loss = self.criterion(reconstruct, batch)
        
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

