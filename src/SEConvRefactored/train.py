import torch.nn as nn
import wandb
import torch
from tqdm import tqdm
from utils import get_device, count_parameters
from DynamicCNN import DynamicCNN

def train(model: DynamicCNN,
          optimizer,
          criterion,
          train_loader,
          val_loader,
          expansion_threshold,
          epochs,
          upgrade_amount,
          initial_lr,
          ) -> dict:
    device = get_device()
    model = model.to(device)
    num_params_initial = count_parameters(model)

    history = {
        "train_loss": [],
        "val_loss": [],
        "train_accuracy": [],
        "val_accuracy": [],
        "num_parameters": []
    }

    wandb.login()
    wandb.init(project="SECNN", mode="online")

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

        model.expand_if_necessary(
            train_loader, expansion_threshold, criterion, upgrade_amount)
        
        optimizer = torch.optim.Adam(model.parameters(), lr=initial_lr)  # Reinitialize optimizer

        # Calculate the scaling factor for the learning rate
        scaling_factor = num_params_initial / (count_parameters(model))

        # Update the learning rate of the optimizer
        optimizer.param_groups[0]['lr'] = initial_lr * scaling_factor

        num_params = count_parameters(model)
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_accuracy'].append(train_accuracy)
        history['val_accuracy'].append(val_accuracy)
        history['num_parameters'].append(num_params)

        wandb.log({
            "train_loss": train_loss,
            "validation_loss": val_loss,
            "train_accuracy": train_accuracy,
            "validation_accuracy": val_accuracy,
            "num_params": num_params
        })

        print(f'Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f}, '
              f'Train Accuracy: {train_accuracy:.2f}%, '
              f'Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%')
    wandb.finish()
    return history
