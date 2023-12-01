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
