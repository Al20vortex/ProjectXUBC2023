import torch.nn as nn
import wandb
import torch
from tqdm import tqdm
from utils import get_device

def train(model,
          train_loader,
          val_loader,
          epochs,
          expansion_threshold) -> dict:
    device = get_device()
    model = model.to(device)

    LEARNING_RATE = 4e-3

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()

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

        wandb.log({
            "train_loss": train_loss,
            "train_accuracy": train_accuracy
        })

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

        if (epoch % 2) == 0:
            model.expand_if_necessary(
                train_loader, expansion_threshold, criterion)
        # model.expand_if_necessary(dataloader = train_loader)

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_accuracy'].append(train_accuracy)
        history['val_accuracy'].append(val_accuracy)

        wandb.log({
            # "train_loss": train_loss,
            "validation_loss": val_loss,
            # "train_accuracy": train_accuracy,
            "validation_accuracy": val_accuracy
        })

        print(f'Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f}, '
              f'Train Accuracy: {train_accuracy:.2f}%, '
              f'Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%')
    wandb.finish()
    return history
