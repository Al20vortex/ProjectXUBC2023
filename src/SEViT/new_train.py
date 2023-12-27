import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms, datasets
from VisionTransformer import VisionTransformer
from CustomDataset import CustomDataset
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import wandb
from utils import get_device
# from positional import PositionalEncoding

wandb.login()
# Your existing code for paths, learning rate, etc.

current_dir = Path("CustomDataset.py").absolute()
parent_path = current_dir.parent.parent.parent.absolute()
parent_path = str(parent_path)
train_path = parent_path+"/llama-duck-ds/train"
validation_path = parent_path + "/llama-duck-ds/val"

lr = 3e-4

device = get_device()  # torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
EPOCHS = 2
BATCH_SIZE = 32


def calculate_accuracy(y_true, y_pred) -> float:
    """
    Calculates the accuracy of the model
    :param y_true: True labels
    :param y_pred: Predicted labels
    :return: Accuracy
    """
    correct = torch.eq(y_true, y_pred).sum().item()
    accuracy = (correct / len(y_pred)) * 100
    return accuracy


def train_2(ml_model, learning_rate=lr, image_size=64):
    """
    Train a Vision Transformer model on a custom dataset.

    Args:
        ml_model (VisionTransformer): The Vision Transformer model to train.
        learning_rate (float): Learning rate for training.
        image_size (int): Size to which images are resized.
    """

    # wandb.init(project="ViT")
    # wandb.config={
    #     "learning_rate": learning_rate,
    #     "epochs": EPOCHS
    # }

    train_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485], std=[0.229])
    ])

    val_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.45], std=[0.229])
    ])
    # ---
    train_dataset = datasets.MNIST(
        root='./mnist_data', train=True, download=True, transform=train_transform)
    validation_dataset = datasets.MNIST(
        root='./mnist_data', train=False, download=True, transform=val_transform)

    # ---

    # train_dataset = datasets.ImageFolder(root=train_path, transform=train_transform)
    # validation_dataset = datasets.ImageFolder(root=validation_path, transform=val_transform)

    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=BATCH_SIZE, shuffle=True)
    validation_loader = DataLoader(
        dataset=validation_dataset, batch_size=BATCH_SIZE, shuffle=False)

    train_losses = []
    validation_losses = []
    train_accuracies = []
    validation_accuracies = []

    print(len(train_dataset), len(validation_dataset))

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(ml_model.parameters(),
                           lr=learning_rate, weight_decay=1e-3)

    ml_model.to(device=device)

    # Training loop
    for epoch in tqdm(range(EPOCHS)):
        ml_model.train()
        total_train, correct_train = 0, 0
        train_loss_epoch = []

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            # Forward pass
            outputs = ml_model(images)
            loss = criterion(outputs, labels)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            _, predicted = torch.max(outputs, dim=1)
            correct_train += torch.eq(labels, predicted).sum().item()
            # print(f"correct_train: {correct_train}\ncorrect_train_type: {type(correct_train)}")
            total_train += labels.size(0)
            train_loss_epoch.append(loss.item())
            # wandb.log({"batch_train_loss": loss.item(),
            #            "batch_train_accuracy": calculate_accuracy(labels, predicted)})

        train_accuracy = 100 * correct_train / total_train
        train_losses.append(np.mean(train_loss_epoch))
        train_accuracies.append(train_accuracy)
        # wandb.log({"epoch_train_loss": np.mean(train_loss_epoch),
        #            "epoch_train_accuracy": train_accuracy})

        # Validation phase
        ml_model.eval()
        total_val, correct_val = 0, 0
        validation_loss_epoch = []
        with torch.no_grad():
            for val_images, val_labels in validation_loader:
                val_images, val_labels = val_images.to(
                    device), val_labels.to(device)

                outputs = ml_model(val_images)
                val_loss = criterion(outputs, val_labels)
                validation_loss_epoch.append(val_loss.item())

                _, val_predicted = torch.max(outputs, dim=1)
                correct_val += torch.eq(val_labels, val_predicted).sum().item()
                print(f"correct_val: {correct_val}")
                total_val += val_labels.size(0)
                print(f"total_val: {total_val}")
                print(f"V_labels {val_labels}")
                print(f"v_preds: {val_predicted}")
                # wandb.log({"batch_val_loss": val_loss.item(),
                #            "batch_val_accuracy": calculate_accuracy(val_labels, val_predicted)})

        val_accuracy = 100 * correct_val / total_val
        validation_losses.append(np.mean(validation_loss_epoch))
        validation_accuracies.append(val_accuracy)
        # wandb.log({"epoch_val_loss": np.mean(validation_loss_epoch),
        #            "epoch_val_accuracy": val_accuracy})
    # wandb.finish()

    return (train_losses, validation_losses), (train_accuracies, validation_accuracies)


# Your model initialization and training call here
if __name__ == "__main__":
    model = VisionTransformer(
        image_size=64,
        patch_size=4,
        embed_dim=24,
        num_layers=8,
        num_heads=8,
        num_classes=10,
        in_channels=1,
    )
    print(model)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(n_params, "n params")
    losses, accuracies = train_2(ml_model=model)
    t_loss, v_loss = losses
    t_acc, v_acc = accuracies
    # print(t_loss, v_loss)
    plt.plot(t_loss, label="Train loss")
    plt.plot(v_loss, label="Validation loss")
    plt.title("Losses over time")
    plt.grid()
    plt.legend()
    plt.show()
    plt.close()

    plt.plot(t_acc, label=" Train accuracy")
    plt.plot(v_acc, label="Validation Accuracy")
    plt.title("Accuracies over time")
    plt.legend()
    plt.grid()
    plt.show()
