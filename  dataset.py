from PIL import Image
import numpy as np
import pandas as pd 
import os
import torch
from torch.utils.data import Dataset







class feature_dataset(Dataset):
    def __init__(self, df):
        self.feature = df.iloc[:, 3:]
        self.label = df['label']

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):

        feature = self.feature.iloc[idx, :]
        label = self.label.iloc[idx]

        # Convert feature DataFrame to tensor
        feature = torch.tensor(feature.values, dtype=torch.float32).squeeze()
        return feature, label
    
        