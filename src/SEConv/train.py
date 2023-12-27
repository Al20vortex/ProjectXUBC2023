import torch
from torch.utils.data import DataLoader
from typing import Tuple, List
# from self_expanding_CNN import SelfExpandingCNN
from new_SeConv import SelfExpandingCNN
import wandb
from tqdm import tqdm
from utils import get_device

import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from torchvision import datasets, transforms


def train_model(model: SelfExpandingCNN,
                train_loader: DataLoader,
                val_loader: DataLoader,
                epochs: int,
                expansion_threshold: float) -> dict:
    """
    Train and validate the SelfExpandingCNN model.

    Args:
        model (SelfExpandingCNN): The neural network model to train.
        train_loader (DataLoader): DataLoader for the training dataset.
        val_loader (DataLoader): DataLoader for the validation dataset.
        epochs (int): Number of epochs to train the model.
        expansion_threshold (float): Threshold for expanding the network.

    Returns:
        dict: Dictionary containing training and validation loss and accuracy per epoch.
    """
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = get_device()
    model = model.to(device)

    optimizer = Adam(model.parameters())
    criterion = CrossEntropyLoss()

    history = {
        "train_loss": [],
        "val_loss": [],
        "train_accuracy": [],
        "val_accuracy": []
    }

    wandb.login()
    wandb.init(project="SelfExpandingCnn", mode="online")

    for epoch in tqdm(range(epochs)):
        model.train()
        train_loss, train_correct, train_total = 0, 0, 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()

        train_loss /= len(train_loader)
        train_accuracy = 100 * train_correct / train_total

        val_loss, val_correct, val_total = 0, 0, 0
        model.eval()
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        val_loss /= len(val_loader)
        val_accuracy = 100 * val_correct / val_total

        model.expand_if_necessary(train_loader, criterion, expansion_threshold)

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_accuracy'].append(train_accuracy)
        history['val_accuracy'].append(val_accuracy)

        wandb.log(history)

        print(f'Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f}, '
              f'Train Accuracy: {train_accuracy:.2f}%, '
              f'Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%')
    wandb.finish()
    return history

# def train_model(model: SelfExpandingCNN,
#                 train_loader: DataLoader,
#                 validation_loader: DataLoader,
#                 criterion: torch.nn.Module,
#                 optimizer: torch.optim.Optimizer,
#                 num_epochs: int,
#                 threshold: float) -> List[Tuple[float, float, float, float]]:
#     """
#     Train and validate the model.
#     """
#     device = get_device()
#     model.to(device)
#     metrics = []

#     wandb.login()
#     wandb.init(project="SelfExpandingConvs", mode="online")

#     for epoch in range(num_epochs):
#         # Training phase
#         model.train()
#         train_loss, train_correct, train_total = 0, 0, 0

#         for inputs, labels in train_loader:
#             inputs, labels = inputs.to(device), labels.to(device)
#             optimizer.zero_grad()

#             outputs = model(inputs)
#             loss = criterion(outputs, labels)
#             loss.backward()
#             optimizer.step()

#             train_loss += loss.item()
#             _, predicted = torch.max(outputs.data, 1)
#             train_total += labels.size(0)
#             train_correct += (predicted == labels).sum().item()

#         train_accuracy = 100. * train_correct / train_total
#         avg_train_loss = train_loss / len(train_loader)
#         wandb.log({
#             "train Accuracy": train_accuracy,
#             "train_loss": avg_train_loss
#         })

#         # Validation phase
#         model.eval()
#         validation_loss, validation_correct, validation_total = 0, 0, 0

#         with torch.no_grad():
#             for inputs, labels in validation_loader:
#                 inputs, labels = inputs.to(device), labels.to(device)
#                 outputs = model(inputs)
#                 loss = criterion(outputs, labels)

#                 validation_loss += loss.item()
#                 _, predicted = torch.max(outputs.data, 1)
#                 validation_total += labels.size(0)
#                 validation_correct += (predicted == labels).sum().item()

#         validation_accuracy = 100. * validation_correct / validation_total
#         avg_validation_loss = validation_loss / len(validation_loader)
#         wandb.log({
#             "Validation_accuracy": validation_accuracy,
#             "validation_loss": avg_validation_loss
#         })

#         # Append metrics for this epoch
#         metrics.append((avg_train_loss, avg_validation_loss,
#                        train_accuracy, validation_accuracy))

#         # Expand network if necessary
#         model.expand_if_necessary(train_loader, criterion, threshold)
#         model.to(device)

#         print(f'Epoch [{epoch+1}/{num_epochs}], '
#               f'Train Loss: {avg_train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%, '
#               f'Validation Loss: {avg_validation_loss:.4f}, Validation Accuracy: {validation_accuracy:.2f}%')

#     return metrics


# def train_model(model: SelfExpandingCNN,
#                 train_loader: DataLoader,
#                 validation_loader: DataLoader,
#                 criterion: torch.nn.Module,
#                 optimizer: torch.optim.Optimizer,
#                 num_epochs: int,
#                 threshold: float) -> List[Tuple[float, float, float, float]]:
#     """
#     Train and validate the model.

#     Args:
#         model (SelfExpandingCNN): The model to train.
#         train_loader (DataLoader): DataLoader for training data.
#         validation_loader (DataLoader): DataLoader for validation data.
#         criterion (torch.nn.Module): Loss function.
#         optimizer (torch.optim.Optimizer): Optimizer.
#         num_epochs (int): Number of training epochs.
#         threshold (float): Threshold for expanding the network.

#     Returns:
#         List[Tuple[float, float, float, float]]: List of tuples containing
#         (train_loss, validation_loss, train_accuracy, validation_accuracy).
#     """

#     device = get_device()
#     wandb.login()
#     wandb.init(project="SelfExpandingConvs", mode="online")
#     metrics = []
#     model.to(device=device)

#     for epoch in tqdm(range(num_epochs)):
#         # Training phase
#         model.train()
#         train_loss, correct, total = 0, 0, 0
#         for inputs, labels in train_loader:
#             inputs, labels = inputs.to(device), labels.to(device)
#             optimizer.zero_grad()
#             outputs = model(inputs)
#             loss = criterion(outputs, labels)
#             loss.backward()
#             optimizer.step()

#             train_loss += loss.item()
#             _, predicted = torch.max(outputs.data, 1)
#             total += labels.size(0)
#             correct += (predicted == labels).sum().item()

#         avg_train_loss = train_loss / len(train_loader)
#         train_accuracy = correct / total
#         wandb.log(
#             {
#                 "train_loss": avg_train_loss,
#                 "train_accuracy": train_accuracy
#             }
#         )

#         # Validation
#         model.eval()
#         validation_loss, correct, total = 0, 0, 0
#         with torch.no_grad():
#             for inputs, labels in validation_loader:
#                 inputs = inputs.to(device)
#                 labels = labels.to(device)

#                 outputs = model(inputs)
#                 loss = criterion(outputs, labels)

#                 validation_loss += loss.item()
#                 _, predicted = torch.max(outputs.data, 1)
#                 total += labels.size(0)
#                 correct += (predicted == labels).sum().item()

#         avg_validation_loss = validation_loss / len(validation_loader)
#         validation_accuracy = correct / total
#         wandb.log(
#             {
#                 "Validation_loss": avg_validation_loss,
#                 "validation_accuracy": validation_accuracy
#             }
#         )

#         # Append metrics
#         metrics.append((avg_train_loss, avg_validation_loss,
#                        train_accuracy, validation_accuracy))

#         # expand if score above threshold
#         model.expand_if_necessary(train_loader, criterion, threshold)
#         print(model)
#     wandb.finish()
#     return metrics


# def adjust_learning_rate(optimizer,
#                          epoch,
#                          initial_lr,
#                          decay_rate,
#                          decay_epoch,
#                          new_layer_lr,
#                          new_layer_epoch,
#                          new_layers_params):
#     base_lr = initial_lr * (decay_rate ** (epoch // decay_epoch))
#     if epoch < new_layer_epoch:
#         lr = new_layer_lr
#     else:
#         lr = base_lr

#     for param_group in optimizer.param_groups:
#         if param_group['params'] == new_layers_params:
#             param_group['lr'] = lr
#         else:
#             param_group['lr'] = base_lr


# def train_model(model: SelfExpandingCNN,
#                 train_loader: DataLoader,
#                 validation_loader: DataLoader,
#                 criterion: torch.nn.Module,
#                 num_epochs: int,
#                 threshold: float) -> List[Tuple[float, float, float, float]]:
#     """
#     Train and validate the model.
#     """
#     device = get_device()
#     print(f"Device: {device}")

#     # Define initial learning rates and decay
#     initial_lr = 0.01
#     new_layer_lr = 0.001
#     new_layer_epoch = 10  # Epoch after which new layers' LR catches up
#     decay_rate = 0.5
#     decay_epoch = 10

#     # Partitioning model parameters
#     all_params = model.parameters()

#     # Define a single optimizer
#     optimizer = torch.optim.Adam(all_params, lr=initial_lr)

#     wandb.login()
#     wandb.init(project="SelfExpandingConvs", mode="online")
#     metrics = []
#     model.to(device=device)

#     for epoch in tqdm(range(num_epochs)):
#         # Training phase
#         model.train()
#         train_loss, correct, total = 0, 0, 0
#         for inputs, labels in train_loader:
#             inputs, labels = inputs.to(device), labels.to(device)
#             optimizer.zero_grad()
#             outputs = model(inputs)
#             loss = criterion(outputs, labels)
#             loss.backward()
#             optimizer.step()

#             train_loss += loss.item()
#             _, predicted = torch.max(outputs.data, 1)
#             total += labels.size(0)
#             correct += (predicted == labels).sum().item()

#         avg_train_loss = train_loss / len(train_loader)
#         train_accuracy = correct / total
#         wandb.log({"train_loss": avg_train_loss,
#                   "train_accuracy": train_accuracy})

#         # Validation phase
#         model.eval()
#         validation_loss, correct, total = 0, 0, 0
#         with torch.no_grad():
#             for inputs, labels in validation_loader:
#                 inputs = inputs.to(device)
#                 labels = labels.to(device)
#                 outputs = model(inputs)
#                 loss = criterion(outputs, labels)

#                 validation_loss += loss.item()
#                 _, predicted = torch.max(outputs.data, 1)
#                 total += labels.size(0)
#                 correct += (predicted == labels).sum().item()

#         avg_validation_loss = validation_loss / len(validation_loader)
#         validation_accuracy = correct / total
#         wandb.log({"Validation_loss": avg_validation_loss,
#                   "validation_accuracy": validation_accuracy})

#         # Append metrics
#         metrics.append((avg_train_loss, avg_validation_loss,
#                        train_accuracy, validation_accuracy))

#         # Expand if necessary
#         model.expand_if_necessary(train_loader, criterion, threshold)

#     wandb.finish()
#     return metrics
