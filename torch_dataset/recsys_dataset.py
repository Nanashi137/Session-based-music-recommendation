import torch
from torch.utils.data import Dataset

import polars as pl
import numpy as np

class RecsysDataset(Dataset):
    def __init__(self, data_path:str):
        super().__init__()

        df = pl.read_csv(data_path).drop("session_id")

        self.sequences = df['sequence'].to_list()
        self.targets = df['target'].to_list()

    def __len__(self):
        return len(self.targets)
    
    def __getitem__(self, idx):
        sequence = (lambda x: [int(i) for i in x])(self.sequences[idx].split("-"))
        target = int(self.targets[idx])

        return sequence, len(sequence), target
