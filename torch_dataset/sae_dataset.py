import torch
from torch.utils.data import Dataset

import polars as pl
import numpy as np

class SAE_dataset(Dataset):
    def __init__(self, df_path: str) -> None:
        super().__init__()

        self.df = pl.read_csv(df_path)

    def __len__(self): 
        return len(self.df)
    
    def __getitem__(self, idx):
        features = self.df[idx].drop("track_id")

        features_vector = torch.tensor(np.array(features, dtype=np.float32))

        return features_vector