import torch


def get_device():
    if torch.cuda.is_available():
        return "cuda"  # for guy who had to sell a kidney for the nvdidia gpu
    elif torch.backends.mps.is_available():
        return "mps"  # for the mac guy (alright I guess)
    else:
        return "cpu"  # for the poor guy with no gpu (sell a kidney dude)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
