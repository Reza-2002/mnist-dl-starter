import torch

def accuracy(predictions, labels):
    _, predicted = torch.max(predictions, 1)
    return (predicted == labels).sum().item() / labels.size(0)
