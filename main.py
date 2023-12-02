import torch
import torchvision
from torch import nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import itertools
from torchsummary import summary
import wandb
import random


# Define our hyperparameters
INPUT_SHAPE = (3, 64, 64)
OUTPUT_SHAPE = (1,)
POSSIBLE_MODULES = [nn.Conv2d, nn.MaxPool2d]  # TODO ADD MORE, like CONV2d TRANSPOSE
TAU = 0.8
EPOCHS = 1000
LEARNING_RATE = 1e-3
BATCH_SIZE = 256

# Set up hyperparameters for each possible module
module_config = {
    "Conv2d": {
        "out_channels": [8, 16, 32, 64, 128],  # Adjust as per your requirement
        "kernel_size": [3, 5, 7, 9],
        "stride": [1, 2],
    },
    "MaxPool2d": {
        "kernel_size": [2, 3],
        "stride": [2],
    }
}


wandb.init(
    # set the wandb project where this run will be logged
    project="ProjectX",
    
    # track hyperparameters and run metadata
    config={
    "learning_rate": LEARNING_RATE,
    "architecture": "ConvClassifier",
    "dataset": "DuckLlama",
    "tau": TAU,
    "epochs": EPOCHS,
    }
)

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
        self.input_layer = nn.Sequential(nn.Conv2d(3, 8, kernel_size=3, stride=1, padding=1), nn.ReLU())
        self.layers = nn.ModuleList()

        # output layer placeholder
        self.out = None
        self.adapt_output_layer(8, INPUT_SHAPE[1], INPUT_SHAPE[2])

    def adapt_output_layer(self, num_feature_maps, x_dim, y_dim):
        self.out = nn.Sequential(
            nn.Flatten(),
            nn.Linear(num_feature_maps * x_dim * y_dim, OUTPUT_SHAPE[0]),
        ).to(device)

    def insert_layer(self, position, module: nn.Module, **kwargs):
        if position < 0 or position > len(self.layers):
            raise ValueError("Invalid position to insert layer")

        new_layer = module(**kwargs).to(device)
        if type(new_layer) == nn.Conv2d:
            # Initialize conv layers with identity kernels to keep output relatively unaffected.
            identity_conv_init(new_layer)
            self.layers.insert(position, new_layer)
            self.layers.insert(position+1, nn.ReLU())  # TODO instead make it choose from some activation choices somehow.
        else:            
            self.layers.insert(position, new_layer)

    
    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, x):
        x = self.input_layer(x)
        for layer in self.layers:
            x = layer(x)
        x = self.out(x)
        return x.view(x.size(0))
    
    def compute_fisher_information(self, dataloader: torch.Tensor, criterion: torch.Tensor) -> dict:
        """
        Computes the fisher matrix

        Args:
            dataloader (torch.Tensor): The dataloader
            criterion (torch.Tensor): The criterion. 

        Returns:
            dict: The fisher information
        """
        fisher_information = {}
        for name, param in self.named_parameters():
            fisher_information[name] = torch.zeros_like(param)

        self.eval()
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device).float(), labels.to(device).float()
            outputs = self(inputs)
            loss = criterion(outputs, labels)

            self.zero_grad()
            loss.backward()

            for name, param in self.named_parameters():
                if param.grad is not None:
                    fisher_information[name] += (param.grad **
                                                    2) / len(dataloader)
        return fisher_information
    
    def compute_natural_expansion_score(self, dataloader: torch.Tensor, criterion: torch.Tensor) -> float:
        """
        Computes the natural expansion score as per the paper SelfExpandinNeuralNetworks

        Args:
            dataloader (torch.Tensor): The dataloader baby
            criterion (torch.Tensor): The criterion baby

        Returns:
            float: The score baby
        """
        fisher_information = self.compute_fisher_information(
            dataloader, criterion)

        natural_grad_approx = 0
        num_params = self.count_parameters()
        for name, param in self.named_parameters():
            if param.grad is not None and name in fisher_information:
                fisher_diag = fisher_information[name]
                natural_grad_approx += (param.grad **
                                        2 / (fisher_diag + 1e-5)).sum()
        return natural_grad_approx.item() / num_params  # TODO should this be averaged out like this?


def identity_conv_init(conv_layer):
    """
    Initialize a Conv2D layer as an identity operation.
    """
    # Assuming the Conv2D layer has the same number of input and output channels
    in_channels = conv_layer.in_channels
    out_channels = conv_layer.out_channels
    kernel_size = conv_layer.kernel_size[0]

    # Check if in_channels and out_channels are equal
    if in_channels != out_channels:
        raise ValueError("Identity initialization requires the same number of input and output channels.")

    # Initialize the weights
    identity_kernel = torch.zeros(out_channels, in_channels, kernel_size, kernel_size)
    for i in range(in_channels):
        if kernel_size % 2 == 1:  # Odd kernel size
            center = kernel_size // 2
            identity_kernel[i, i, center, center] = 1
        else:
            raise NotImplementedError("Identity initialization not implemented for even kernel sizes.")

    # Set the weights and disable bias
    conv_layer.weight.data = identity_kernel.to(device).float()
    conv_layer.bias.data = torch.zeros(out_channels).to(device).float()

      
# Define transformations for the dataset
transform = transforms.Compose([
    transforms.Resize((INPUT_SHAPE[1], INPUT_SHAPE[2])),  # Resize images
    transforms.ToTensor()
    # Add more transformations if necessary
])

# Load the datasets
train_dataset = datasets.ImageFolder('llama-duck-ds/train', transform=transform)
val_dataset = datasets.ImageFolder('llama-duck-ds/val', transform=transform)

# Data loaders
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

model = NeuralNet().to(device)

# TODO params needs to be useful information based off we decide to expand. For now we will simply expand every second epoch
def need_to_expand(model: NeuralNet, loss_fun):
    """Checks if the model can benefit from expansion based on natural gradients"""
    n_expansion_score = model.compute_natural_expansion_score(train_loader, loss_fun)
    print("Natural Expansion Score: ", n_expansion_score)
    return n_expansion_score > TAU  # TODO find better threshold
    # if epoch % 2 == 0:
    #     return True
    # else:
    #     return False


# TODO params needs to be useful information for determining if the module will be beneficial
# 	Must also take into account the layer we are adding the module to.
def get_module_benefit(params, module):
    """Returns a benefit score for the module"""
    # stub
    return 0.5

def select_best_action(benefits):
    # Sort the benefits and get the best one
    benefits.sort(key=lambda x: x[0], reverse=True)
    return benefits[0]  # Returns the tuple (benefit_score, position, module_type, config)              

def update_padding_for_module(module_type, config_dict):
    """Returns the module_type and config_dicts with padding equal to:
        - same for nstrided Conv2d
        - Half the image size for strided Conv2d or Pooling
    """

    # TODO IN THE FUTURE EXPAND SUPPORT OF MORE MODULES
    if module_type.__name__ == 'Conv2d':
        # Convolution padding calculation
        if config_dict.get('stride', 1) == 1:
            kernel_size = config_dict.get('kernel_size', 0)
            config_dict['padding'] = (kernel_size - 1) // 2
        elif config_dict.get('stride', 1) == 2:
            kernel_size = config_dict.get('kernel_size', 0)
            config_dict['padding'] = (kernel_size - 2) // 2
            config_dict['padding'] = max(config_dict['padding'], 0)

    elif module_type.__name__ == 'MaxPool2d':
        # Pooling padding calculation
        kernel_size = config_dict.get('kernel_size', 0)
        if kernel_size % 2 == 0:
            config_dict['padding'] = 0  # or minimal padding as required
        else:
            config_dict['padding'] = (kernel_size - 1) // 2
    return config_dict

def calculate_insertion_benefits(model, input_shape, possible_modules, module_config):
    benefits = []
    x = torch.rand((1, *input_shape)).to(device)  # Dummy input for shape inference
    x = model.input_layer(x)
    
    for i in range(len(model.layers) + 1):
        for module_type in possible_modules:
            # Generate all combinations of configurations for the module
            configs = list(itertools.product(*module_config[module_type.__name__].values()))
            for config in configs:
                # Create a dictionary of arguments for the module constructor
                config_dict = {k: v for k, v in zip(module_config[module_type.__name__].keys(), config)}
                # print(f"config_dict: {config_dict}")
                if module_type.__name__ == 'Conv2d':
                    # Here we set the correct in_channels for convolution modules because theres always only one possibility based on the prev layer's output shape
                    config_dict['in_channels'] = x.shape[1]

                # Must also calculate the padding based on the stride and the kernel size.
                config_dict = update_padding_for_module(module_type, config_dict)
                
                # Insert hypothetical module
                hypothetical_module = module_type(**config_dict).to(device)

                # print(f"hypothetical_module: {hypothetical_module}")
                hypothetical_x = hypothetical_module(x)
                # print(f"hypothetical_x shape: {hypothetical_x.shape}")
                # Calculate benefit score (stub)
                benefit_score = get_module_benefit({}, hypothetical_module)
                
                # Store the benefit along with position and configuration
                benefits.append((benefit_score, i, module_type, config_dict))
        
        # Forward pass through the next actual layer
        if i < len(model.layers):
            x = model.layers[i](x)
    return benefits

def train():
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loss_fun = nn.BCEWithLogitsLoss()

    model.train()  # Set the model to training mode
    correct_train = 0
    total_train = 0

    for epoch in range(0, EPOCHS):
        for input, target in train_loader:  
            input, target = input.to(device).float(), target.to(device).float()
            optimizer.zero_grad()
            output = model(input)
            loss = loss_fun(output, target)
            predicted = torch.sigmoid(output).round()  # Getting the binary predictions
            correct_train += (predicted == target).sum().item()
            total_train += target.size(0)
            loss.backward()
            optimizer.step()
        train_acc = 100 * correct_train / total_train

        # calculate validation scores
        model.eval()  # Set the model to evaluation mode
        correct_val = 0
        total_val = 0
        val_loss_total = 0
        for val_input, val_target in val_loader:  
            val_input, val_target = val_input.to(device).float(), val_target.to(device).float()
            optimizer.zero_grad()
            val_output = model(val_input)
            val_loss = loss_fun(val_output, val_target)
            val_loss_total += val_loss.item()
            predicted_val = torch.sigmoid(val_output).round()
            correct_val += (predicted_val == val_target).sum().item()
            total_val += val_target.size(0)

        val_acc = 100 * correct_val / total_val
        avg_val_loss = val_loss_total / len(val_loader)
        num_params = model.count_parameters()
        wandb.log({"Train Loss": loss, "Train Accuracy": train_acc, "Val Loss": avg_val_loss, "Val Accuracy": val_acc, "Model Size (Params)": num_params})
        
        print(f'Epoch: {epoch} Train Loss: {loss.item()} Train Acc: {train_acc:.2f}% Val Loss: {avg_val_loss:.4f} Val Acc: {val_acc:.2f}% Number of Parameters: {num_params}')

        if epoch % 10 == 0:
            print("Model Summary:")
            summary(model, input_size=INPUT_SHAPE)


        # Example usage in the train function:
        if need_to_expand(model, loss_fun):
            # Calculate benefits for all possible insertions
            benefits = calculate_insertion_benefits(model, INPUT_SHAPE, POSSIBLE_MODULES, module_config)

            # Select the best action
            _, position, best_module, best_config = select_best_action(benefits)

            # Insert the new layer
            model.insert_layer(position, best_module, **best_config)

            # Update the next layer after that one as necessary
            # TODO For this we need to adapt that next layer to accept the number of feature maps outputted by the newly added layer.
            # Can we do that without deleting the parameters we already learned and recreating it? 
            # Maybe we can copy the learned kernels and biases instead, and let the newly added ones to make shapes match be initalized randomly, then they get learned all together.

train()
