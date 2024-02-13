import torch
import torch.nn as nn
from torch.utils.data import DataLoader
# from train import train
from utils import get_device
from torchvision import transforms, datasets
BATCH_SIZE = 512
image_size = 32

train_transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[
                         0.2023, 0.1994, 0.2010])
])

val_transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[
                         0.2023, 0.1994, 0.2010])
])

cifar_train = datasets.CIFAR10(
    root="datasets/cifar_train", train=True, transform=train_transform, download=True)
cifar_test = datasets.CIFAR10(
    root="datasets/cifar_test", train=False, transform=val_transform, download=True)

cifar_train_loader = DataLoader(
    cifar_train, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
cifar_test_loader = DataLoader(
    cifar_test, batch_size=BATCH_SIZE, shuffle=False, drop_last=True)
