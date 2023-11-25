import torch
import torchvision
from torch import nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Define our hyperparameters
INPUT_SHAPE = (64, 64)
OUTPUT_SHAPE = (1,)
POSSIBLE_MODULES = [nn.Conv2d, nn.MaxPool2d]  # Added MaxPool2d for diversity

# Set up hyperparameters for each possible module
module_config = {
    "Conv2d": {
        "in_channels": [3, 64, 128],  # Example channel sizes
        "out_channels": [64, 128, 256],  # Adjust as per your requirement
        "kernel_size": [3, 5],
        "stride": [1, 2],
        "padding": [0, 1]
    },
    "MaxPool2d": {
        "kernel_size": [2, 3],
        "stride": [2],
        "padding": [0, 1]
    }
}

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")

device =  get_device()

class NeuralNet(nn.Module):
    def __init__(self):      
        super(NeuralNet, self).__init__()

        # input layer
        self.input_layer = nn.Sequential(nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1), nn.ReLU())
        self.layers = nn.ModuleList()

        # output layer placeholder
        self.out = None
        self.adapt_output_layer(64, 64, 64)

    def adapt_output_layer(self, num_feature_maps, x_dim, y_dim):
        self.out = nn.Sequential(
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(num_feature_maps * x_dim * y_dim, OUTPUT_SHAPE[0]),
            nn.Sigmoid()
        )

    def add_layer(self, module: nn.Module, **kwargs):
        self.layers.append(module(**kwargs))

    def forward(self, x):
        x = self.input_layer(x)
        for layer in self.layers:
            x = layer(x)
        x = self.out(x)
        return x.view(x.size(0))

# TODO params needs to be useful information based off we decide to expand. For now we will simply expand every second epoch
def need_to_expand(params, epoch):
    """Checks if the model can benefit from expansion based on natural gradients"""
    if epoch % 2 == 0:
        return True
    else:
        return False


# TODO params needs to be useful information for determining if the module will be beneficial
# 	Must also take into account the layer we are adding the module to.
def get_module_benefit(params, module):
    """Returns a benefit score for the module"""

    # stub
    return 0.5                    

# Define transformations for the dataset
transform = transforms.Compose([
    transforms.Resize(INPUT_SHAPE),  # Resize images
    transforms.ToTensor()
    # Add more transformations if necessary
])

# Load the datasets
train_dataset = datasets.ImageFolder('llama-duck-ds/train', transform=transform)
val_dataset = datasets.ImageFolder('llama-duck-ds/val', transform=transform)

# Data loaders
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

def train():
    model = NeuralNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fun = nn.BCELoss()

    for epoch in range(1000):
        for input, target in train_loader:  
            input, target = input.to(device).float(), target.to(device).float()
            optimizer.zero_grad()
            output = model(input)
            loss = loss_fun(output, target)
            loss.backward()
            optimizer.step()

        print(f'Epoch: {epoch} Loss: {loss.item()}')

        if need_to_expand({}, epoch):

            # Same as the forward function but we get the last thing in model layers 
            # Determine the shape of the last layer
            x = torch.rand((1, 3, 64, 64)).to(device).float()  # Dummy input for shape inference
            x = model.input_layer(x)
            for layer in model.layers:
                x = layer(x)
            shape = x.shape  # Shape before the output layer

            # TODO: Define logic to choose best module and its configuration
            # As an example, adding a Conv2d layer
            best_module = nn.Conv2d
            best_module_config = {"in_channels": shape[1], "out_channels": 128, "kernel_size": 3, "stride": 1, "padding": 1}

            model.add_layer(best_module, **best_module_config)
            model.adapt_output_layer(128, shape[2], shape[3])  # Update output layer

train()
