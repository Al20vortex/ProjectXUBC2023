import torch.nn as nn
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from pathlib import Path
import os

current_dir = Path("CustomDataset.py").absolute()
parent_path = current_dir.parent.parent.parent.absolute()
parent_path = str(parent_path)

train_path = parent_path + "/llama-duck-ds/train"
valid_path = parent_path + "/llama-duck-ds/val"

train_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

train_dataset = datasets.ImageFolder(root=train_path, transform=train_transform)
validation_dataset = datasets.ImageFolder(root=valid_path, transform=val_transform)

train_loader = DataLoader(dataset=train_dataset, batch_size=32, )

if __name__ == "__main__":
    print(len(train_dataset))
    print(len(validation_dataset))
    print(train_dataset)
    print(parent_path)
    print(len(os.listdir(valid_path+"/duck"))+len(os.listdir(valid_path+"/llama")))
