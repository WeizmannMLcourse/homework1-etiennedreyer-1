
import torch
import numpy as np
import torch.nn as nn


class Net(nn.Module):
    def __init__(self):
        super().__init__()

        self.layer1 = ...
    
    def forward(self,x):
        
        ...

        return out