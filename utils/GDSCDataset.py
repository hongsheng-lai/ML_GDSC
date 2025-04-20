# dataset.py

import torch
from torch.utils.data import Dataset
import pandas as pd
from dataloader import load_gdsc, prepare_features

class GDSCDataset(Dataset):
    def __init__(self):
        excluded_columns = ['LN_IC50', 'AUC', 'Z_SCORE', 'DRUG_ID', 'COSMIC_ID', 'DRUG_NAME', 'CELL_LINE_NAME']
        df = load_gdsc(excluded_columns=excluded_columns)   # With Drop NaN & Exclude Outlier with IQR
        X_dummy, y = prepare_features(df, encode_dummies=True)
        self.features = torch.from_numpy(X_dummy.values).float()
        self.labels   = torch.from_numpy(y.values).float().unsqueeze(1)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]
