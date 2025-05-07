import torch

class RecsysCollateFunction: 
    def __init__(self, padding_idx):
        self.padding_idx = padding_idx

    def __call__(self, batch):
        seqs, seq_lens, targets = zip(*batch)

        longest_len = max(seq_lens)
        [seq.extend([self.padding_idx]*(longest_len-len(seq))) for seq in seqs] # padding

        seqs = torch.tensor(seqs, dtype=torch.int32)
        seq_lens = torch.tensor(seq_lens, dtype=torch.int32)
        targets = torch.tensor(targets, dtype=torch.int32)

        return seqs, seq_lens, targets


if __name__ == "__main__":
    seqs = [[1,2,3,4],[1,2,3],[1]]
    seq_lens = [4, 3, 1]

    longest_len = max(seq_lens)
    [seq.extend([3773]*(longest_len-len(seq))) for seq in seqs] # padding

    seqs = torch.tensor(seqs, dtype=torch.int32)
    seq_lens = torch.tensor(seq_lens, dtype=torch.int32)

    print(seqs)
    print(seq_lens)
    






