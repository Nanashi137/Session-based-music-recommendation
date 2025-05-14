import torch 
import torch.nn as nn 
import torch.nn.functional as F

class contrastive_encoder(nn.Module):
    def __init__(self, input_dim, embedding_dim):
        super().__init__()

        self.input_dim = input_dim
        self.embedding_dim = embedding_dim  

        self.encoder = nn.Sequential(
                nn.Linear(in_features=self.input_dim, out_features=32, bias=True),
                nn.BatchNorm1d(32),
                nn.ReLU(),
                nn.Dropout(p=0.2),
                nn.Linear(in_features=32, out_features=32),
                nn.BatchNorm1d(32),
                nn.ReLU(),
                nn.Dropout(p=0.2),
                nn.Linear(in_features=32, out_features=self.embedding_dim),
            )

    def forward(self, x):
        embedding = self.encoder(x)
        embedding = F.normalize(embedding, p=2, dim=1, eps=1e-12)
        return embedding
    
    def n_trainable_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
if __name__ == "__main__":
    x = torch.randn(32, 11)
    model = contrastive_encoder(input_dim=11, embedding_dim=64)
    print(model.n_trainable_params())