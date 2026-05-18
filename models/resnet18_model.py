import torch
from torchvision import models

def create_model(num_classes):

    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

    for param in model.parameters():
        param.requires_grad = False

    model.fc = torch.nn.Linear(
        model.fc.in_features,
        num_classes
    )
    return model

