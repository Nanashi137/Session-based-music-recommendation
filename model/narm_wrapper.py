import pytorch_lightning as L
from pytorch_lightning.utilities.grads import grad_norm

from torch.optim.lr_scheduler import StepLR
import torch 
import torch.nn as nn
import numpy as np

from .narm import NARM 


class NARMWrapper(L.LightningModule): 
    def __init__(self, embedding_dim, hidden_size, n_gru_layers, n_items, padding_idx: int=None, embedding_matrix_path: str=None, lr: float=1e-3, k: int=10):
        super().__init__()
        self.save_hyperparameters()

        self.lr = lr
        self.k = k
        self.criterion = nn.CrossEntropyLoss(reduction="mean", ignore_index= padding_idx if padding_idx is not None else n_items)

        self.embedding_matrix = self._load_embedding_matrix(embedding_matrix_path=embedding_matrix_path) if embedding_matrix_path else None
        self.model = NARM(embedding_dim=embedding_dim, hidden_size=hidden_size, n_gru_layers=n_gru_layers, n_items=n_items, padding_idx=padding_idx, embedding_matrix=self.embedding_matrix)

    def _load_embedding_matrix(self, embedding_matrix_path):
        torch_embedding_matrix = torch.tensor(np.load(file=embedding_matrix_path).astype(np.float32))
        return torch_embedding_matrix


    def forward(self, seq, seq_len):
        return self.model(seq, seq_len)

    def inference(self, sequence_embs: torch.Tensor, k: int=20): 
        '''
            Input:
                sequence_embs: sequence of track embedding within session, shape: lxd where l is sequence length, d is embedding dimension
            Output: 
                track_idices: indices of top-k track to recommend 
        '''
        
        gru_out, hidden = self.model.gru(sequence_embs, self.model.h0.view(self.model.n_gru_layers, self.model.hidden_size))

        ht = hidden[-1] # batch_size, hidden_dim

        q1 = self.model.A2(gru_out.view(-1, self.model.hidden_size)).view(gru_out.shape) # seq_length,hidden_dim
        q2 = self.model.A1(ht).unsqueeze(dim=0) # 1, hidden_dim
        alpha = self.model.v_t(torch.sigmoid(q1 + q2)).squeeze(dim=1) # seq_length
        alpha = torch.softmax(alpha, dim=0)
        attention_output = torch.einsum("i, ij -> j", alpha, gru_out) # hidden_dim

        c_t = torch.cat([ht, attention_output], dim=0) # 2*hidden_dim
        
        item_embs = self.model.emb(torch.arange(start=0, end=self.model.n_items, step=1).to(device = sequence_embs.device)) # n_items, embedding_dim
        decoded_item_embs = self.model.B(item_embs) # n_items, 2*hidden_dim
        logits = torch.matmul(c_t, decoded_item_embs.T) # batch_size, n_items

        return torch.topk(logits, k=k).indices

    def training_step(self, batch, batch_idx):

        seqs, seq_lens, targets = batch
        logits = self(seqs, seq_lens)

        loss = self.criterion(logits, targets)

        self.log("train_loss", loss, prog_bar=True, on_epoch=True, logger=True)
        
        return loss
        

    def validation_step(self, batch, batch_idx):
        loss = self._shared_eval(batch)
        self.log("val_loss", loss, prog_bar=True, on_epoch=True, logger=True)

        metrics = self.evaluation_metric(batch, "Validation")
        self.log_dict(metrics, prog_bar=True)

        return {"val_loss": loss}
    
    def test_step(self, batch, batch_idx):
        loss = self._shared_eval(batch)
        self.log_dict({"test_loss": loss})

        metrics = self.evaluation_metric(batch, "Test")
        self.log_dict(metrics, prog_bar=True)
    
    def _shared_eval(self, batch):
        seqs, seq_lens, targets = batch
        logits = self(seqs, seq_lens)

        loss = self.criterion(logits, targets)
        
        return loss

    def evaluation_metric(self, batch, set):
        seqs, seq_lens, targets = batch
        logits = self(seqs, seq_lens)

        # Get top k items
        topk = torch.topk(logits, self.k, dim=1).indices  # batch_size, k
        targets = targets.view(-1, 1) # batch_size , 1

        # Hit rate
        hits = (topk == targets).float()
        hr = hits.sum(dim=1).mean()

        # MRR
        rr = hits/torch.arange(1, self.k+1, device=logits.device).float()
        mrr = rr.sum(dim=1).mean()

        # nDCG
        discount = torch.log2(torch.arange(2, self.k + 2, device=logits.device).float())
        dcg = (hits / discount).sum(dim=1)
        ndcg = dcg.mean()

        return {
            f"{set} HR{self.k}": hr.item(),
            f"{set} MRR{self.k}": mrr.item(),
            f"{set} nDCG{self.k}": ndcg.item()
        }

    def on_before_optimizer_step(self, optimizer):
        # Compute the L2-norm for each layer
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
