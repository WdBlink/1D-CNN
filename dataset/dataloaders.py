import torch
import os
from torch.utils.data import Dataset


class MyDataset(Dataset):
    def __init__(self, data, label):
        self.data = data
        self.label = label

    def __getitem__(self, index):
        return (torch.tensor(self.data[index], dtype=torch.float), torch.tensor(self.label[index], dtype=torch.long))

    def __len__(self):
        return len(self.data)


class AFDataset(torch.utils.data.Dataset):
    def __init__(self, root):
        self.root = root
        self.all_data = root['data']
        self.all_label = root['label']

    def __getitem__(self, idx):
        # load images and bbox
        data = self.all_data[idx]
        label = self.all_label[idx]
        return data, label

    def __len__(self):
        return len(self.all_data)
