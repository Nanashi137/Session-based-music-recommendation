# Sparse auto-encoder 
import torch.nn as nn 
import numpy as np
import torch

class SAE(nn.Module):
    def __init__(self, dim_list: list[int]):
        super(SAE, self).__init__()

        self.dim_list = dim_list
        
        # Encoder
        encoder_modules = []
        for i in range(1, len(self.dim_list)-1):
            encoder_modules.append(nn.Linear(self.dim_list[i-1], self.dim_list[i], bias=True))
            encoder_modules.append(nn.ReLU())
        encoder_modules.append(nn.Linear(in_features=self.dim_list[-2], out_features=self.dim_list[-1], bias=True))
        encoder_modules.append(nn.Sigmoid())

        self.encoder = nn.Sequential(*encoder_modules) 
        
        # Decoder
        decoder_modules = []
        decoder_modules.append(nn.Linear(in_features=self.dim_list[-1], out_features=self.dim_list[-2], bias=True))
        for i in range(len(self.dim_list)-2, 0, -1):
            decoder_modules.append(nn.ReLU())
            decoder_modules.append(nn.Linear(self.dim_list[i], self.dim_list[i-1], bias=True))

        self.decoder = nn.Sequential(*decoder_modules)

    def n_params(self):
        return sum(p.numel() for p in self.encoder.parameters() if p.requires_grad) + sum(p.numel() for p in self.decoder.parameters() if p.requires_grad)

    def encode(self, x):
        return self.encoder(x)

    def decode(self, latent_embedding):
        return self.decoder(latent_embedding)
    
    def forward(self, x):
        latent_embedding = self.encode(x)
        decoded_embedding = self.decode(latent_embedding)
        return decoded_embedding, latent_embedding
    
    # When not using lightning 
    def save_checkpoint(self, path, optimizer=None, epoch=None, loss=None): 
        checkpoint = {
            'model_state_dict': self.state_dict(),
            'optimizer_state_dict': optimizer.state_dict() if optimizer is not None else None,
            'epoch': epoch,
            'loss': loss #loss.item()
        }
        torch.save(checkpoint, path)
        print(f"Checkpoint saved to {path}")

    def load_checkpoint(self, path, optimizer=None, device='cpu'):
        checkpoint = torch.load(path, map_location=device)
        self.load_state_dict(checkpoint['model_state_dict'])
        
        if optimizer is not None and checkpoint['optimizer_state_dict'] is not None:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        epoch = checkpoint.get('epoch', 0)
        loss = checkpoint.get('loss', float('inf'))
        print(f"Checkpoint loaded from {path}")
        return epoch, loss

# Debug
if __name__ == "__main__":
    dim_list = [11, 8, 4]
    x = torch.randn(32, 11)
    sae = SAE(dim_list=dim_list)
    print(sae)

