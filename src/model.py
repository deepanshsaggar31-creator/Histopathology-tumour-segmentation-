import torch.nn as nn
from torchvision import models

def get_resnet18(pretrained=True, num_classes=2):
    """
    Builds a ResNet-18 model with a custom classification head.
    """
    if pretrained:
        weights = models.ResNet18_Weights.IMAGENET1K_V1
    else:
        weights = None
        
    model = models.resnet18(weights=weights)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, num_classes)
    return model
