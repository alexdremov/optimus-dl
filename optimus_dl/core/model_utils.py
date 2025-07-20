import torch


def get_num_parameters(model: torch.nn.Module):
    params = set()
    for param in model.parameters():
        params.add(param)
    return sum(param.numel() for param in params)
