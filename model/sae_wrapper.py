import pytorch_lightning as L
from pytorch_lightning.utilities.grads import grad_norm

from torch.optim.lr_scheduler import StepLR
import torch 
import torch.nn as nn

from .sae import SAE

class SAE_wrapper(L.LightningModule): 
    def __init__(self, dim_list: list, lr: float=1e-3, sparsity_parameter: float=0.05, sparsity_weight: float=1e-3):
        super().__init__()
        self.save_hyperparameters()

        self.model = SAE(dim_list=dim_list)
        self.lr = lr
        self.sparsity_parameter = sparsity_parameter
        self.sparsity_weight = sparsity_weight

        self.criterion = nn.MSELoss()

    def forward(self, x):
        return self.model(x)
    
    def encode(self, x): 
        return self.model.encode(x)
    
    def decode(self, x): 
        return self.model.decode(x)
    
    def kl_divergence(self, rho_hat):
        rho = self.sparsity_parameter
        return torch.sum(
            rho * torch.log(rho / (rho_hat + 1e-9)) + 
            (1 - rho) * torch.log((1 - rho) / (1 - rho_hat + 1e-9))
        )

    def training_step(self, batch, batch_idx):
        reconstruct, z = self.forward(batch)
        rho_hat = torch.mean(z, dim=0)
        sparsity_loss = self.kl_divergence(rho_hat)
        reconstruct_loss = self.criterion(reconstruct, batch)
        loss = reconstruct_loss + self.sparsity_weight*sparsity_loss
        self.log("train_reconstruct_loss", reconstruct_loss, prog_bar=True, on_epoch=True, logger=True)
        self.log("train_sparsity_loss", sparsity_loss, prog_bar=True, on_epoch=True, logger=True)
        self.log("train_loss", loss, prog_bar=True, on_epoch=True, logger=True)
        
        return loss

    def validation_step(self, batch, batch_idx):
        reconstruct_loss, sparsity_loss = self._shared_eval(batch=batch)
        loss = reconstruct_loss + self.sparsity_weight*sparsity_loss
        self.log("val_reconstruct_loss", reconstruct_loss, prog_bar=True, on_epoch=True, logger=True)
        self.log("val_sparsity_loss", sparsity_loss, prog_bar=True, on_epoch=True, logger=True)
        self.log("val_loss", loss, prog_bar=True, on_epoch=True, logger=True)
        return {"val_loss": loss}
    
    def test_step(self, batch, batch_idx):
        reconstruct_loss, _ = self._shared_eval(batch=batch)
        self.log_dict({"test_reconstruct_loss": reconstruct_loss})
    
    def _shared_eval(self, batch):
        reconstruct, z = self.forward(batch)
        rho_hat = torch.mean(z, dim=0)
        sparsity_loss = self.kl_divergence(rho_hat)
        reconstruct_loss = self.criterion(reconstruct, batch)
        
        
        return reconstruct_loss, sparsity_loss

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

