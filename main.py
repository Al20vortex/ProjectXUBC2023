import torch
import torchvision
import torch
from torch import nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


# Define our hyperparameters
INPUT_SHAPE = (64, 64)
OUTPUT_SHAPE = (1,)
POSSIBLE_MODULES = [nn.Conv2d, ]

# TODO set up these hyper params as config options for each possible module
module_config = {
    "Conv2d": {
        "kernel_size": [1, 3, 5],
        "padding": [0, 1, 2],
        "activation": [nn.ReLU, nn.LeakyReLU]
    },
}

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
    def __init__(self):		
        super(NeuralNet, self).__init__()
        
        # input layer
        self.input_layer = nn.Sequential(nn.Conv2d(3, 3, kernel_size=3, stride=1, padding=1), nn.ReLU())
        self.layers = nn.ModuleList()

        # Currently using a preset output layer which lowers the resolution from 64x64 down to 1x1
        #  TODO This needs to change in the future to be more adaptive/nonrigid
        self.out = nn.Sequential(
            nn.Conv2d(3, 1, kernel_size=3, stride=1, padding=1),   # TODO we need to make this infer somehow so that it connects to how many feature maps were in the previous layer. 
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(1*64*64, 1),
            nn.Sigmoid()
        )

    def adapt_output_layer(self, num_feature_maps, x_dim, y_dim):
        self.out = nn.Sequential(
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(num_feature_maps*x_dim*y_dim, OUTPUT_SHAPE),
            nn.Sigmoid()
        )

    def add_layer(self, module: nn.Module):
        self.layers.append(module())
    
    def forward(self, x):
        x = self.input_layer(x)
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

    # TODO complete the code to load the dataset
    model = NeuralNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fun = nn.BCELoss()

    for epoch in range(1000):
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
                    # TODO in here we should keep track of the shape of the last layer before the output.
                    #   This is in order to ensure the layer we're adding can accept inputs of this shape

                    # also for each laeyr, we should have a list of possible initialization options

                    # check the benefit score of adding this module
                    benefit_scores[j] = get_module_benefit({}, module)
                
                best_module = POSSIBLE_MODULES[torch.argmax(benefit_scores)]

                # work out the parameters for that module
                # like if its a conv layer we need to figure out the dims for that spot
                # model.add_layer(best_module) TODO DOESNT WORK

train()