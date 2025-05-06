import torch
from torch.utils.data import Dataset

import polars as pl
import numpy as np

class Recsys_dataset(Dataset):
    def __init__(self, data_path:str):
        super().__init__()

        self.df = pl.read_csv(data_path)

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        
