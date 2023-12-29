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

BATCH_SIZE = 32
EPOCHS = 20
print(emnist_train[0][0].shape)
train_loader = DataLoader(emnist_train, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(emnist_val, batch_size=BATCH_SIZE, shuffle=False)

cifar_train_loader = DataLoader(
    cifar_train, batch_size=BATCH_SIZE, shuffle=True)
cifar_test_loader = DataLoader(
    cifar_test, batch_size=BATCH_SIZE, shuffle=False)

channels_list = [1, 8, 8]
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
    images, labels = next(iter(cifar_test_loader))
    print(f"cifar_shape: {images[0].shape}")
    imshow(images[0])
    # print(cifar_test[0])
    # print(cifar_test_loader)

    # ___
    # model = DynamicCNN(channels_list, 10, 0.2)
    # history = train(
    #     model=model,
    #     train_loader=cifar_train_loader,
    #     val_loader=cifar_test_loader,
    #     expansion_threshold=1.5,
    #     epochs=EPOCHS
    # )
    # ___
    # print("****")
    # print(model.convs)
    # print("***")
    # scores = []

    # num_convs = len(model.convs)

    # conv_block_indices = []
    # for index, module in enumerate(model.convs):
    #     if isinstance(module, ConvBlock):
    #         conv_block_indices.append(index)

    # print(conv_block_indices)
    # print(f"Before addition:\n{model.convs} ")
    # for index in conv_block_indices:
    #     # temp_model = copy.deepcopy(model)
    #     # print(f"CONVS: {model.convs[index]}")
    #     # model.convs[index].add_layer()
    #     print(f"expanding at index {index}")
    #     model.expand(index)
    #     print(model.convs)

    # for index in conv_block_indices:
    #     print(f"expanding again at index {index}")
    #     model.expand(index)
    #     print(model.convs)

    # print("temp stuff")
    # print(model.convs)
    # del temp_model

    # print(f"after addition: ")
    # print(model.convs)
    # print(model)
    # print(f"Scores: {scores}")
    # optimal_index = torch.argmax(torch.Tensor(scores))
    # print(
    #     f"Optimal Index: {optimal_index}, type: {type(optimal_index.item())}")
    # a = torch.tensor([1, 2, 1.])
    # print(torch.argmax(a).item())
