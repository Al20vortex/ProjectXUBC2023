import os
import torch
import torch.nn as nn
from PIL import Image
import numpy as np

# Checking image dimensions
parent_dir = os.getcwd()
image_path = parent_dir+"/llama-duck-ds/train/duck/00AONW8ADLQD.jpg"
pil_image = Image.open(image_path)
image = np.array(pil_image)

if __name__ == "__main__":
    print(image.shape)
