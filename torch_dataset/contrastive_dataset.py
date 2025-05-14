import torch
from torch.utils.data import Dataset

import polars as pl
import numpy as np

class ContrastiveDataset(Dataset):
    def __init__(self, data_path:str, id2features_matrix_path:str):
        super().__init__()

        df = pl.read_csv(data_path)
        self.id2features = np.load(id2features_matrix_path).astype(np.float32)
        self.item_id = df['track_idx'].to_list()
        self.cluster_id = df['cluster_id'].to_list()

    def __len__(self):
        return len(self.item_id)
    
    def __getitem__(self, idx):
        features_vector = torch.tensor(self.id2features[self.item_id[idx]])
        normalized_features_vector = features_vector/features_vector.norm(p=2)
        label = torch.tensor(self.cluster_id[idx])

        return normalized_features_vector, label
