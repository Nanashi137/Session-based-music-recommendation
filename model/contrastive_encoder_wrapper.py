import pytorch_lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR

from .contrastive_encoder import contrastive_encoder

from pytorch_lightning.utilities.grads import grad_norm
from pytorch_metric_learning import miners, losses, distances, reducers


class ContrastiveEncoderWrapper(L.LightningModule):
    def __init__(self, input_dim,embedding_dim: int, learning_rate: float=0.01, margin: float=0.5):
        super().__init__()
        self.save_hyperparameters()

        self.learning_rate = learning_rate 
        self.embedding_dim = embedding_dim 
        self.input_dim = input_dim
        self.margin = margin

        self.model = contrastive_encoder(input_dim=self.input_dim, embedding_dim=self.embedding_dim)

    def forward(self, X): 
        return self.model(X)
    
    def training_step(self, batch, batch_idx): 
        self.model.train()
        X, Labels = batch

        Embeddings = self(X)

        distance = distances.CosineSimilarity()
        reducer = reducers.ThresholdReducer(low=0)
        
        miner = miners.TripletMarginMiner(margin=self.margin, type_of_triplets="hard")

        loss_fn = losses.TripletMarginLoss(margin=self.margin, distance= distance, reducer=reducer)
        
        hard_pairs = miner(Embeddings, Labels)
        loss = loss_fn(Embeddings, Labels, hard_pairs)

        self.log("train_loss", loss, prog_bar=True, on_epoch=True, logger=True)

        return loss
    
    def validation_step(self, batch, batch_idx):
        self.model.eval()
        X, Labels = batch

        Embeddings = self(X)

        distance = distances.CosineSimilarity()
        reducer = reducers.ThresholdReducer(low=0)
        
        miner = miners.TripletMarginMiner(margin=self.margin, type_of_triplets="hard")

        loss_fn = losses.TripletMarginLoss(margin=self.margin, distance= distance, reducer=reducer)
        
        hard_pairs = miner(Embeddings, Labels)
        loss = loss_fn(Embeddings, Labels, hard_pairs)
        self.log("val_loss", loss, prog_bar=True, on_epoch=True, logger=True)
        return {"val_loss": loss}



    def on_before_optimizer_step(self, optimizer):
        # Compute the L2 norm for each layer
        # If using mixed precision, the gradients are already unscaled here
        norms = grad_norm(self.model, norm_type=2)
        self.log_dict(norms)


    def configure_optimizers(self):
        params = filter(lambda p: p.requires_grad, self.parameters())
        optimizer = torch.optim.AdamW(
            params, lr=self.learning_rate, weight_decay=0.001
        )
        scheduler = StepLR(
            optimizer, step_size=5, gamma=0.8
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
            },
        }




# Debug 
if __name__ == "__main__":
    embedding_size = 32
    model = ContrastiveEncoderWrapper(input_dim=11,
                         embedding_dim= embedding_size,
                         margin=1.5,
                         learning_rate=0.001
                         )
    
    a = torch.randn(32, 11)
    print(model(a))