import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from networks import *
from networks.utils import emnist_train, emnist_val
from networks.utils import train, get_device

device = get_device()

BATCH_SIZE = 32
EPOCHS = 20
print(emnist_train[0][0].shape)
train_loader = DataLoader(emnist_train, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(emnist_val, batch_size=BATCH_SIZE, shuffle=False)

channels_list = [1, 8, 8]
n_classes = 10
model = DynamicCNN(channels_list=channels_list, n_classes=10)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


if __name__ == "__main__":

    model = DynamicCNN(channels_list, 10, 0.2)
    history = train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        expansion_threshold=2,
        epochs=EPOCHS
    )
    print(model)
