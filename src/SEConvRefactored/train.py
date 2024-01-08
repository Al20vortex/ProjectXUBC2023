import torch.nn as nn
import wandb
import torch
from tqdm import tqdm
from utils import get_device, count_parameters
from DynamicCNN import DynamicCNN

# L1_REG = 1e-4
# L1_REG = 1e-5
L1_REG = 5e-6

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
    cooldown = 0
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, verbose=True)

    history = {
        "train_loss": [],
        "val_loss": [],
        "train_accuracy": [],
        "val_accuracy": [],
        "num_parameters": [],
        "learning_rate": []
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
            
            # Calculate L1 Regularization
            l1_reg = torch.tensor(0.).to(device)
            for param in model.parameters():
                l1_reg += torch.norm(param, 1)

            # Combine loss with L1 regularization
            loss = criterion(outputs, labels)
            loss_ = loss + L1_REG * l1_reg
            loss_.backward()
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
        scheduler.step(val_loss)
        val_loss /= len(val_loader)
        val_accuracy = 100 * val_correct / val_total

        current_lr = optimizer.param_groups[0]['lr']

        if cooldown <= 0:
            expanded = model.expand_if_necessary(
                train_loader, expansion_threshold, criterion, upgrade_amount)
            if expanded:
                optimizer = torch.optim.Adam(model.parameters(), lr=current_lr)  # Reinitialize optimizer
                scheduler.optimizer = optimizer  # Maintain reference to correct optimizer
                cooldown = 10
                print(model.convs)
        cooldown -= 1

        num_params = count_parameters(model)
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_accuracy'].append(train_accuracy)
        history['val_accuracy'].append(val_accuracy)
        history['num_parameters'].append(num_params)
        history['learning_rate'].append(current_lr)
        wandb.log({
            "train_loss": train_loss,
            "validation_loss": val_loss,
            "train_accuracy": train_accuracy,
            "validation_accuracy": val_accuracy,
            "num_params": num_params,
            "learning_rate": current_lr
        })

        print(f'Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f}, '
              f'Train Accuracy: {train_accuracy:.2f}%, '
              f'Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%')
    wandb.finish()
    return history
