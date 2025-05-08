import torch

class RecsysCollateFunction: 
    def __init__(self, padding_idx):
        self.padding_idx = padding_idx

    def __call__(self, batch):
        seqs, seq_lens, targets = zip(*batch)

        longest_len = max(seq_lens)
        [seq.extend([self.padding_idx]*(longest_len-len(seq))) for seq in seqs] # padding

        seqs = torch.tensor(seqs)
        seq_lens = torch.tensor(seq_lens)
        targets = torch.tensor(targets)

        return seqs, seq_lens, targets








