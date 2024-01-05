import torch
from torch import nn

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def get_device():
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"

# Debugging tool
def check_network_consistency(model):
    prev_out_channels = None
    for block in model.convs:
        for layer in block.convs:
            if isinstance(layer, nn.Conv2d):
                if prev_out_channels is not None and prev_out_channels != layer.in_channels:
                    raise ValueError(f"Channel mismatch: previous layer's output {prev_out_channels}, current layer's input {layer.in_channels}")
                prev_out_channels = layer.out_channels

    # Check the transition to the one_by_one_conv layer
    if prev_out_channels != model.one_by_one_conv[0].in_channels:
        raise ValueError(f"Channel mismatch at one_by_one_conv: expected {prev_out_channels}, found {model.one_by_one_conv[0].in_channels}")