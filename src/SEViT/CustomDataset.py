import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import Dataset
import os
from pathlib import Path
import random
random.seed(1)

current_dir = Path("CustomDataset.py").absolute()
parent_path = current_dir.parent.parent.parent.absolute()
parent_path = str(parent_path)
print(current_dir, "current dir")
print(f"parent path: {parent_path}")


class CustomDataset(Dataset):
    """
    A custom dataset class
    """

    def __init__(self, class_A_path: str, class_B_path: str, transforms=None) -> None:
        """
        Creates the custom dataset
        Args:
            class_A_path: The path to class A. in my case, duck
            class_B_path: The path to class B. In my case llama
            transforms:
        """
        super().__init__()
        self.transforms = transforms
        self.class_A_images = [os.path.join(class_A_path, img) for img in os.listdir(class_A_path)]
        self.class_B_images = [os.path.join(class_B_path, img) for img in os.listdir(class_B_path)]
        self.all_images = [(img, 0) for img in self.class_A_images] + \
                          [(img, 1) for img in self.class_B_images]
        random.shuffle(self.all_images)

    def __len__(self):
        return len(self.all_images)

    def __getitem__(self, idx):
        img_path, label = self.all_images[idx]
        image = Image.open(img_path).convert('RGB')

        if self.transforms is not None:
            image = self.transforms(image)

        return image, label


if __name__ == "__main__":
    A = parent_path + "/llama-duck-ds/train/duck"
    B = parent_path + "/llama-duck-ds/train/llama"
    dataset = CustomDataset(A, B)
    print(len(dataset))
    print(dataset[0])
