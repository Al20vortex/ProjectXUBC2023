import os
import torch
import torch.nn as nn
from PIL import Image
import numpy as np
from pathlib import Path


def calculate_accuracy(y_true, y_pred) -> float:
    """
    Calculates the accuracy of the model
    :param y_true: True labels
    :param y_pred: Predicted labels
    :return: Accuracy
    """
    correct = torch.eq(y_true, y_pred).sum().item()
    accuracy = (correct / len(y_pred)) * 100
    return accuracy



current_dir = Path("CustomDataset.py").absolute()
parent_path = current_dir.parent.parent.parent.absolute()
parent_path = str(parent_path)
# Checking image dimensions
parent_dir = os.getcwd()
image_path = parent_path+"/llama-duck-ds/train/llama/0A9GP5ZX0C0W.jpg"
pil_image = Image.open(image_path)
image = np.array(pil_image)
att = nn.MultiheadAttention(embed_dim=8, num_heads=8)
a = torch.tensor([[1.,2], [0.1, 0.01]])
# pose_embed = nn.Embedding(n)
if __name__ == "__main__":
    y = torch.rand(10, 2)
    z = torch.rand(10, 2)
    _, y = torch.max(y, dim=1)
    _, z = torch.max(z, dim=1)
    a = calculate_accuracy(y, z)
    b = calculate_accuracy(z, y)
    print(y)
    print(z)
    print(a)
    print(b)