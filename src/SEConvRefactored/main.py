import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from DynamicCNN import *
from train import train
from utils import get_device
from torchvision import transforms, datasets

device = get_device()

BATCH_SIZE = 512
EPOCHS = 1000
LEARNING_RATE = 2e-3
UPGRADE_AMT = 4 # BEST
DROPOUT = 0.1
image_size = 32

train_transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])
])

val_transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])
])

cifar_train = datasets.CIFAR10(
    root="./cifar_train", train=True, transform=train_transform, download=True)
cifar_test = datasets.CIFAR10(
    root="./cifar_test", train=False, transform=val_transform, download=True)

cifar_train_loader = DataLoader(
    cifar_train, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
cifar_test_loader = DataLoader(
    cifar_test, batch_size=BATCH_SIZE, shuffle=False, drop_last=True)

channels_list = [3, 16, 16, 16]
n_classes = 10
model = DynamicCNN(channels_list=channels_list,
                   n_classes=n_classes, 
                   image_size=image_size, 
                   pooling_stride=2,
                   dropout=DROPOUT).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
criterion = nn.CrossEntropyLoss()

history = train(
    model=model,
    optimizer=optimizer,
    criterion=criterion,
    train_loader=cifar_train_loader,
    val_loader=cifar_test_loader,
    expansion_threshold=2.0,
    epochs=EPOCHS,
    upgrade_amount=UPGRADE_AMT,
    initial_lr=LEARNING_RATE,
)
