import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from networks import *
from networks.utils import emnist_train, emnist_val, cifar_test, cifar_train
from networks.utils import train, get_device
import copy
import numpy as np
import matplotlib.pyplot as plt

device = get_device()

BATCH_SIZE = 400
EPOCHS = 400
# print(emnist_train[0][0].shape)
train_loader = DataLoader(emnist_train, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(emnist_val, batch_size=BATCH_SIZE, shuffle=False)

cifar_train_loader = DataLoader(
    cifar_train, batch_size=BATCH_SIZE, shuffle=True)
cifar_test_loader = DataLoader(
    cifar_test, batch_size=BATCH_SIZE, shuffle=False)

channels_list = [3, 16, 16, 32]
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
    model = DynamicCNN(channels_list, 10, dropout=0.2)
    # print(model.convs[0])
    history = train(
        model=model,
        train_loader=cifar_train_loader,
        val_loader=cifar_test_loader,
        expansion_threshold=1.,
        epochs=EPOCHS
    )
