import torch 
import torch.nn as nn 

class NARM(nn.Module):
    def __init__(self, embedding_dim, hidden_size, gru_layers, ) -> None:
        super(NARM, self).__init__()

        self.hidden_size = hidden_size
        self.embedding_dim = embedding_dim
        self.gru_layers = gru_layers

        self.emb = self._init_embedding(self)
        self.emb_dropout = nn.Dropout(p=0.2)

        # GRU 
        self.gru_layers = nn.GRU(input_size=embedding_dim, hidden_size=self.hidden_size, num_layers=self.gru_layers, batch_first=True) 
        # Local encoder 
        



        # Glocal encoder
