from torch.utils.data import DataLoader, random_split
from torchvision import transforms, datasets

image_size = 28
BATCH_SIZE = 32
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
train_dataset = datasets.MNIST(
    root='./mnist_data', train=True, download=True, transform=train_transform)
validation_dataset = datasets.MNIST(
    root='./mnist_data', train=False, download=True, transform=val_transform)

train_loader = DataLoader(dataset=train_dataset,
                          batch_size=BATCH_SIZE, shuffle=True)
validation_loader = DataLoader(
    dataset=validation_dataset, batch_size=BATCH_SIZE, shuffle=False)

if __name__ == "__main__":
    for i in [train_loader, validation_loader]:
        print(len(i))
