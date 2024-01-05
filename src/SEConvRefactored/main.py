import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from DynamicCNN import *
from train import train
from utils import get_device
import copy
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms, datasets

device = get_device()

BATCH_SIZE = 512
EPOCHS = 100
LEARNING_RATE = 4e-3
UPGRADE_FACTOR = 1.5

image_size = 32

train_transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=0.5, std=0.25)
])

val_transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean=0.5, std=0.25)
])

cifar_train = datasets.CIFAR10(
    root="./cifar_train", train=True, transform=train_transform, download=True)
cifar_test = datasets.CIFAR10(
    root="./cifar_test", train=False, transform=val_transform, download=True)

cifar_train_loader = DataLoader(
    cifar_train, batch_size=BATCH_SIZE, shuffle=True)
cifar_test_loader = DataLoader(
    cifar_test, batch_size=BATCH_SIZE, shuffle=False)

channels_list = [3, 8, 8]
n_classes = 10
model = DynamicCNN(channels_list=channels_list, n_classes=n_classes).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
criterion = nn.CrossEntropyLoss()

# TODO create a config object where user can set the initial conditions
config = {
    "use_pooling": True,
    "use_strided_conv": False,
    "channels_list": channels_list
}

history = train(
    model=model,
    optimizer=optimizer,
    criterion=criterion,
    train_loader=cifar_train_loader,
    val_loader=cifar_test_loader,
    expansion_threshold=2.0,
    epochs=EPOCHS,
    upgrade_factor = UPGRADE_FACTOR
)
