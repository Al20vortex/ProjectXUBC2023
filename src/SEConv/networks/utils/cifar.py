import torch.nn as nn
import torch
from torch.utils.data import DataLoader, random_split
from torchvision import transforms, datasets

image_size = 32
train_transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485], std=[0.229])
])

val_transform = transforms.Compose([

    transforms.Resize((image_size, image_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.45], std=[0.229])
])
cifar_train = datasets.CIFAR10(
    root="../../cifar_train", train=True, transform=train_transform, download=True)
cifar_test = datasets.CIFAR10(
    root="../../cifar_test", train=False, transform=val_transform, download=True)
# print(cifar_train)
