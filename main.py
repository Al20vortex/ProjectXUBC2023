import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from models import DynamicCNN
from utils import train_model
from utils import get_device
from torchvision import transforms, datasets
from dataloaders import cifar_test_loader, cifar_train_loader

device = get_device()

EPOCHS = 1000
LEARNING_RATE = 2e-3
UPGRADE_AMT = 4  # BEST
DROPOUT = 0.1
image_size = 32
channels_list = [3, 16, 16, 16]
n_classes = 10
model = DynamicCNN(channels_list=channels_list,
                   n_classes=n_classes,
                   image_size=image_size,
                   pooling_stride=2,
                   dropout=DROPOUT).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
criterion = nn.CrossEntropyLoss()

history = train_model(
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
