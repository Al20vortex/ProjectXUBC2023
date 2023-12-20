import torch
from torch.utils.data import DataLoader
from typing import Tuple, List
from self_expanding_CNN import SelfExpandingCNN
import wandb
from tqdm import tqdm


def train_model(model: SelfExpandingCNN,
                train_loader: DataLoader,
                validation_loader: DataLoader,
                criterion: torch.nn.Module,
                optimizer: torch.optim.Optimizer,
                num_epochs: int,
                threshold: float) -> List[Tuple[float, float, float, float]]:
    """
    Train and validate the model.

    Args:
        model (SelfExpandingCNN): The model to train.
        train_loader (DataLoader): DataLoader for training data.
        validation_loader (DataLoader): DataLoader for validation data.
        criterion (torch.nn.Module): Loss function.
        optimizer (torch.optim.Optimizer): Optimizer.
        num_epochs (int): Number of training epochs.
        threshold (float): Threshold for expanding the network.

    Returns:
        List[Tuple[float, float, float, float]]: List of tuples containing 
        (train_loss, validation_loss, train_accuracy, validation_accuracy).
    """
    wandb.login()
    wandb.init(project="SelfExpandingConvs")
    metrics = []

    for epoch in tqdm(range(num_epochs)):
        # Training phase
        model.train()
        train_loss, correct, total = 0, 0, 0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        avg_train_loss = train_loss / len(train_loader)
        train_accuracy = correct / total
        wandb.log(
            {
                "train_loss": avg_train_loss,
                "train_accuracy": train_accuracy
            }
        )

        # Validation
        model.eval()
        validation_loss, correct, total = 0, 0, 0
        with torch.no_grad():
            for inputs, labels in validation_loader:
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                validation_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        avg_validation_loss = validation_loss / len(validation_loader)
        validation_accuracy = correct / total
        wandb.log(
            {
                "Validation_loss": avg_validation_loss,
                "validation_accuracy": validation_accuracy
            }
        )

        # Append metrics
        metrics.append((avg_train_loss, avg_validation_loss,
                       train_accuracy, validation_accuracy))

        # expand if score above threshold
        model.expand_if_necessary(train_loader, criterion, threshold)
        print(model)
    wandb.finish()
    return metrics


def adjust_learning_rate(optimizer,
                         epoch,
                         initial_lr,
                         decay_rate,
                         decay_epoch,
                         new_layer_lr,
                         new_layer_epoch,
                         new_layers_params):
    """"""
    base_lr = initial_lr * (decay_rate ** (epoch // decay_epoch))
    if epoch < new_layer_epoch:
        lr = new_layer_lr
    else:
        lr = base_lr

    for param_group in optimizer.param_groups:
        if param_group['params'] == new_layers_params:
            param_group['lr'] = lr
        else:
            param_group['lr'] = base_lr


def train_model(model: SelfExpandingCNN,
                train_loader: DataLoader,
                validation_loader: DataLoader,
                criterion: torch.nn.Module,
                num_epochs: int,
                threshold: float) -> List[Tuple[float, float, float, float]]:
    """
    Train and validate the model.
    """

    # Define initial learning rates and decay
    initial_lr = 0.01
    new_layer_lr = 0.001
    new_layer_epoch = 10  # Epoch after which new layers' LR catches up
    decay_rate = 0.5
    decay_epoch = 10

    # Partitioning model parameters
    all_params = model.parameters()

    # Define a single optimizer
    optimizer = optim.Adam(all_params, lr=initial_lr)

    wandb.login()
    wandb.init(project="SelfExpandingConvs")
    metrics = []

    for epoch in tqdm(range(num_epochs)):
        # Training phase
        model.train()
        train_loss, correct, total = 0, 0, 0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        avg_train_loss = train_loss / len(train_loader)
        train_accuracy = correct / total
        wandb.log({"train_loss": avg_train_loss,
                  "train_accuracy": train_accuracy})

        # Validation phase
        model.eval()
        validation_loss, correct, total = 0, 0, 0
        with torch.no_grad():
            for inputs, labels in validation_loader:
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                validation_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        avg_validation_loss = validation_loss / len(validation_loader)
        validation_accuracy = correct / total
        wandb.log({"Validation_loss": avg_validation_loss,
                  "validation_accuracy": validation_accuracy})

        # Append metrics
        metrics.append((avg_train_loss, avg_validation_loss,
                       train_accuracy, validation_accuracy))

        # Adjust learning rates
        new_layers_params = model.new_layers.parameters() if epoch < new_layer_epoch else []
        adjust_learning_rate(optimizer, epoch, initial_lr, decay_rate,
                             decay_epoch, new_layer_lr, new_layer_epoch, new_layers_params)

        # Expand if necessary
        model.expand_if_necessary(train_loader, criterion, threshold)

    wandb.finish()
    return metrics
