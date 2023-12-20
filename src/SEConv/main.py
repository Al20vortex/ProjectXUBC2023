import torch
import torch.nn as nn
from train import train_model
from data_preprocess import train_loader, validation_loader, BATCH_SIZE
# from self_expanding_CNN import SelfExpandingCNN
from new_SeConv import SelfExpandingCNN


EPOCHS = 20
channels_list = [1, 32, 32]
n_classes = 10
model = SelfExpandingCNN(channels_list, n_classes)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


if __name__ == "__main__":
    print(model)
    metrics = train_model(
        model=model,
        train_loader=train_loader,
        optimizer=optimizer,
        validation_loader=validation_loader,
        criterion=criterion,
        num_epochs=EPOCHS,
        threshold=1.007
    )
    print(metrics)
