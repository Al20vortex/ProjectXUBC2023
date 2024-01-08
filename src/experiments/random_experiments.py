import os
import torch
import torch.nn as nn
from PIL import Image
import numpy as np
from pathlib import Path


a = nn.ModuleList()
print(a)
conv1 = nn.Conv2d(in_channels=3, out_channels=4, kernel_size=3)
conv2 = nn.Conv2d(4, 7, kernel_size=3)
a.insert(0, conv1)
print(a)
a.insert(0, conv2)
print(a)
d = [1, 2, 31]
print(*d)
e = [12, *d]
print(e)
