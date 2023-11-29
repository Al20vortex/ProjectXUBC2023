import torch
import torch.nn as nn
from visionTransformer import VisionTransformer
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms, datasets
from customDataset import CustomDataset
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

current_dir = Path("CustomDataset.py").absolute()
parent_path = current_dir.parent.parent.parent.absolute()
parent_path = str(parent_path)
duck_path = parent_path + "/llama-duck-ds/train/duck"
llamma_path = parent_path + "/llama-duck-ds/train/llama"

lr = 1e-4

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
EPOCHS = 50
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


def train(ml_model,
          class_A_path,
          class_B_path,
          learning_rate=lr,
          image_size=64
          ):
    """
    Train a Vision Transformer model on a custom dataset.

    Args:
        image_size:
        learning_rate:
        ml_model (VisionTransformer): The Vision Transformer model to train.
        class_A_path (str): Path to the directory containing Class A images.
        class_B_path (str): Path to the directory containing Class B images.
    """

    # Transforms
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    train_losses = []
    validation_losses = []

    train_accuracies = []
    validation_accuracies = []

    # Dataset and DataLoader
    dataset = CustomDataset(class_A_path=class_A_path,
                            class_B_path=class_B_path, transforms=transform)
    train_size = int(0.8 * len(dataset))
    validation_size = len(dataset) - train_size
    train_dataset, validation_dataset = random_split(
        dataset=dataset, lengths=[train_size, validation_size])
    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=BATCH_SIZE, shuffle=True)
    validation_loader = DataLoader(
        dataset=validation_dataset, batch_size=BATCH_SIZE, shuffle=False)
    print(len(dataset))
    print(len(train_loader), len(validation_loader), "Loaders")
    print(len(train_dataset), len(validation_dataset))
    # exit()
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Device configuration
    ml_model.to(device)

    # Training loop
    for epoch in tqdm(range(EPOCHS)):
        train_loss_epoch = []
        validation_loss_epoch = []
        total_train = 0
        correct_train = 0
        ml_model.train()

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            # print(f"labels: {labels}")

            # Forward pass
            outputs = ml_model(images)
            # print(outputs)
            # print(outputs.shape, labels.shape)
            loss = criterion(outputs, labels)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            _, predicted = torch.max(outputs, dim=1)

            train_accuracy = calculate_accuracy(
                y_true=labels, y_pred=predicted)
            correct_train += train_accuracy

            train_loss_epoch.append(loss.item())

        # np.mean(correct_train)#correct_train / len(train_dataset)  # total_train
        train_accuracy = correct_train/len(train_loader)
        train_accuracies.append(train_accuracy)
        train_losses.append(np.mean(train_loss_epoch))

        ml_model.eval()
        with torch.no_grad():
            total_val = 0
            correct_val = 0

            for images, labels in validation_loader:
                images, labels = images.to(device), labels.to(device)

                preds = ml_model(images)
                val_loss = criterion(preds, labels)
                validation_loss_epoch.append(val_loss.item())
                # Calculate validation accuracy
                _, predicted = torch.max(preds, dim=1)

                validation_acc = calculate_accuracy(
                    y_true=labels, y_pred=predicted)
                correct_val += validation_acc

            val_accuracy = correct_val / len(validation_loader)  # total_val
            validation_accuracies.append(val_accuracy)
            validation_losses.append(np.mean(validation_loss_epoch))
    return (train_losses, validation_losses), (train_accuracies, validation_accuracies)


if __name__ == "__main__":
    model = VisionTransformer(
        image_size=64,
        patch_size=4,
        embed_dim=16,
        num_layers=8,
        num_heads=8,
        num_classes=2,
        in_channels=3,
    )
    print(model)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(n_params, "n params")
    losses, accuracies = train(
        ml_model=model, class_A_path=duck_path, class_B_path=llamma_path)
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
