import torch
from torch import nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


# Define our hyperparameters
INPUT_SHAPE = (64, 64)
OUTPUT_SHAPE = (1,)
POSSIBLE_MODULES = [nn.Conv2d]

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")

# Use the function to set the device
device = get_device()
print(f"Using device: {device}")

class NeuralNet(nn.Module):
    def __init__(self, initial_module_list):		
        super(NeuralNet, self).__init__()
        self.layers = nn.ModuleList(initial_module_list)


        # Currently using a preset output layer which lowers the resolution from 64x64 down to 1x1
        #   TODO This needs to change in the future to be more adaptive/nonrigid
        self.out = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 1, kernel_size=3, stride=1, padding=1),
            nn.AdaptiveAvgPool2d((1, 1)),  # Resize spatial dimensions to 1x1
            nn.Flatten(),  # Flatten the output
        )

    def add_layer(self, module: nn.Module):
        self.layers.append(module())
    
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        x = self.out(x)
        x = x.view(x.size(0))
        return x

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
    transforms.Grayscale(),
    transforms.ToTensor(),
    # Add more transformations if necessary
])

# Load the datasets
train_dataset = datasets.ImageFolder('llama-duck-ds/train', transform=transform)
val_dataset = datasets.ImageFolder('llama-duck-ds/val', transform=transform)

# Data loaders
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)


def train():

    # complete the code to load the dataset
    initial_layer = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, stride=1, padding=1)

    model = NeuralNet([initial_layer]).to(device)
    optimizer = torch.optim.Adamax(model.parameters(), lr=1e-4)
    loss_fun = nn.BCEWithLogitsLoss()

    for epoch in range(100):
        for input, target in train_loader:  # Iterate over your data
            input, target = input.to(device).float(), target.to(device).float()
            optimizer.zero_grad()
            output = model(input)
            loss = loss_fun(output, target)
            loss.backward()
            optimizer.step()

        print(f'Epoch: {epoch} Loss: {loss}')
        
        # perform some mechanism to determine whether or not to add 
        # check if we need to add a module/layer/neuron
        if need_to_expand({}, epoch):
            # see where we're adding the layer in our model
            for i, layer in enumerate(model.layers):
                # crappy name, rename to something else 
                benefit_scores = torch.zeros(len(POSSIBLE_MODULES))

                for j, module in enumerate(POSSIBLE_MODULES):
                    # check the benefit score of adding this module
                    benefit_scores[j] = get_module_benefit({}, module)
                
                best_module = POSSIBLE_MODULES[torch.argmax(benefit_scores)]

                # work out the parameters for that module
                # like if its a conv layer we need to figure out the dims for that spot
                # model.add_layer(best_module) TODO DOESNT WORK

train()