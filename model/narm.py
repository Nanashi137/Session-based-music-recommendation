import torch 
import torch.nn as nn 
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class NARM(nn.Module):
    def __init__(self, embedding_dim, hidden_size, n_gru_layers, n_items, padding_idx: int=None, embedding_matrix: torch.Tensor=None) -> None:
        super().__init__()

        self.hidden_size = hidden_size
        self.n_items = n_items
        self.embedding_dim = embedding_dim
        self.n_gru_layers = n_gru_layers
        self.padding_idx = n_items if not padding_idx else padding_idx

        self.emb = nn.Embedding(num_embeddings=n_items+1, embedding_dim=self.embedding_dim, padding_idx=self.padding_idx)
        if embedding_matrix is not None:
            self.emb.weight = nn.Parameter(embedding_dim)
            self.emb.weight.requires_grad = False
        
        # GRU 
        self.gru = nn.GRU(input_size=embedding_dim, hidden_size=self.hidden_size, num_layers=self.n_gru_layers, batch_first=True) 
        self.h0 = torch.nn.Parameter(torch.zeros(size=(self.n_gru_layers, 1, self.hidden_size)))

        # Local encoder 
        self.A1 = nn.Linear(in_features=self.hidden_size, out_features=self.hidden_size, bias=False)
        self.A2 = nn.Linear(in_features=self.hidden_size, out_features=self.hidden_size, bias=False)
        self.v_t = nn.Linear(in_features=self.hidden_size, out_features=1, bias=False)

        # Global encoder

        # Decoder
        self.B = nn.Linear(in_features=self.embedding_dim, out_features=2*self.hidden_size)

    def forward(self, sequences, lengths): 
        lengths = lengths.to(dtype=torch.long).cpu()
        # GRU layers
        hidden = self.init_hidden(sequences.shape[0]).to(device = sequences.device)
        embs = self.emb(sequences)
        embs = pack_padded_sequence(embs, lengths, batch_first=True, enforce_sorted=False)
        gru_out, hidden = self.gru(embs, hidden)
        gru_out, _ = pad_packed_sequence(gru_out, batch_first=True, total_length=sequences.shape[1])

        # Global encoder output 
        ht = hidden[-1] # batch_size, hidden_dim

        # Local encoder output 
        q1 = self.A2(gru_out.view(-1, self.hidden_size)).view(gru_out.shape) # batch_size,seq_length,hidden_dim
        q2 = self.A1(ht).unsqueeze(dim=1) #batch_size, 1, hidden_dim

        alpha = self.v_t(torch.sigmoid(q1 + q2)).squeeze(dim=2) # batch,seq_length

        mask = (sequences != self.padding_idx)
        alpha[~mask] = float('-inf')
        alpha = torch.softmax(alpha, dim=1)

        attention_output = torch.einsum("bi, bij -> bj", alpha, gru_out) # batch_size, hidden_dim

        # Session's representation
        c_t = torch.cat([ht, attention_output], dim=1) #batch_size, 2*hidden_dim
        
        # Decoder 
        item_embs = self.emb(torch.arange(start=0, end=self.n_items, step=1).to(device = sequences.device)) # n_items, embedding_dim
        decoded_item_embs = self.B(item_embs) # n_items, 2*hidden_dim

        logits = torch.matmul(c_t, decoded_item_embs.T) # batch_size, n_items

        return logits


    def init_hidden(self, batch_size):
        return self.h0.expand(-1, batch_size, -1).contiguous()
    
    def n_trainable_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

if __name__ == "__main__":
    a = NARM(embedding_dim=11, hidden_size=11, n_gru_layers=1, n_items=6)
    t = torch.tensor([[0, 2, 3, 6, 6], [1, 2, 3, 4, 6]])
    l = torch.tensor([2, 2])
    targets = torch.tensor([2, 3])
    logits = a(t, l)


    loss_fn = nn.CrossEntropyLoss(reduction="mean")
    
    print(logits)
    print(loss_fn(logits, targets))
