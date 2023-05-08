
from torch.utils.data import Dataset
import torch
import h5py

class CustomDataset(Dataset):

    def __init__(self, path, Nmax=None):
        
        with h5py.File(path,'r') as f:
            self.images = torch.FloatTensor(f['images'][:Nmax])
            self.labels = torch.LongTensor(f['ans'][:Nmax])

    def __len__(self):
       
        return len(self.labels)

    def __getitem__(self, idx):
        
        x = self.images[idx]
        x = x.flatten()
        y = self.labels[idx]
    
        return x , y