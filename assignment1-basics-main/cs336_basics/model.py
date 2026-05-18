import torch.nn as nn
import torch

class Linear(nn.Module):
    def __init__(self, in_features: int, out_features: int, device: torch.device = None, type: torch.dtype=None):
        self.in_features = in_features
        self.out_features = out_features
        self.device = device
        self.type = self.type
