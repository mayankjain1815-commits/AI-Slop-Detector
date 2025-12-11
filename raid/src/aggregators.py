import torch

def all_or_nothing(probabilities: torch.Tensor) -> float:
    p_true = torch.prod(probabilities)
    p_false = torch.prod(1 - probabilities)
    P = p_true / (p_true + p_false)

    return P.item()

def take_max(probabilities: torch.Tensor) -> float:
    return probabilities.max().item()

def take_min(probabilities: torch.Tensor) -> float:
    return probabilities.min().item()

def take_mean(probabilities: torch.Tensor) -> float:
    return probabilities.mean().item()
