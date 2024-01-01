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
model = DynamicCNN(channels_list=channels_list, n_classes=10)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


def imshow(img):
    img = img / 2 + 0.5  # unnormalize if necessary
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


if __name__ == "__main__":
    model = DynamicCNN(channels_list, 10, 0.2)
    history = train(
        model=model,
        train_loader=cifar_train_loader,
        val_loader=cifar_test_loader,
        expansion_threshold=0.8,  # 1.5,
        epochs=EPOCHS
    )
