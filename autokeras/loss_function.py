import torch


def classification_loss(prediction, target):
    return torch.nn.CrossEntropyLoss()(prediction, target.long())


def regression_loss(prediction, target):
    return torch.nn.MSELoss()(prediction, target.float())
