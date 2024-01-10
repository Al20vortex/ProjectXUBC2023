import torch
from torch import nn

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def get_device():
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"