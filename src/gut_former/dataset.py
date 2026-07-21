import torch
from torch.utils.data import Dataset


class BacteriaDataset(Dataset):
    def __init__(self, taxonomy, pathways):
        self.taxonomy = torch.tensor(taxonomy.values).float().unsqueeze(-1)
        self.pathways = torch.tensor(pathways.values).float().unsqueeze(-1)

    def __len__(self):
        return len(self.taxonomy)

    def __getitem__(self, idx):
        return self.taxonomy[idx], self.pathways[idx]