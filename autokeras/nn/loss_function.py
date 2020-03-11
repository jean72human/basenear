import torch


def classification_loss(prediction, target):
    #target = target.argmax(1)
    return torch.nn.CrossEntropyLoss()(prediction, target.long())


def regression_loss(prediction, target):
    return torch.nn.MSELoss()(prediction, target.float())


def binary_classification_loss(prediction, label):
    return torch.nn.BCELoss()(prediction, label)
